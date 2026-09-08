# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.utils.checkpoint as checkpoint
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin

from .attention import flash_attention

__all__ = ['WanModel']


def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.amp.autocast('cuda', enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta,
                        torch.arange(0, dim, 2).to(torch.float64).div(dim)))
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@torch.amp.autocast('cuda', enabled=False)
def qkv_rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(
            seq_len, n, -1, 2))
        freqs_i = torch.cat([
            freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ],
                            dim=-1).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()

@torch.amp.autocast('cuda', enabled=False)
def mk_rope_apply(x, grid_fuv, freqs):
    if x.size(1) == 0: # 如果memory为空则直接返回
        return x.float()
    
    n, c = x.size(2), x.size(3) // 2
    
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)
    
    output = []
    for i , fuv in enumerate(grid_fuv):
        mem_len = fuv.size(0)
        
        x_i =torch.view_as_complex(x[i, :mem_len].to(torch.float64).reshape(
            mem_len, n, -1, 2
        ))
        
        f = fuv[:, 0]
        u = fuv[:, 1]
        v = fuv[:, 2]
        
        freq_t = freqs[0][f]
        freq_h = freqs[1][v]
        freq_w = freqs[2][u]
        
        freqs_i = torch.cat(
            [freq_t, freq_h, freq_w],
            dim=-1
        )
        
        # 增加 head 维
        freqs_i =freqs_i.unsqueeze(1)
        
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i,mem_len:]])

        output.append(x_i)
        
    return torch.stack(output).float()


@torch.amp.autocast('cuda', enabled=False)
def _normalize_K(K, h, w):
    """
    K: [B, 3, 3]
    h, w: [B], (image_height, image_width)
    """
    
    out = torch.zeros_like(K)
    
    out[:, 0, 0] = K[:, 0, 0] / w
    out[:, 1, 1] = K[:, 1, 1] / h

    out[:, 0, 2] = K[:, 0, 2] / w - 0.5
    out[:, 1, 2] = K[:, 1, 2] / h - 0.5

    out[:, 2, 2] = 1.0
    
    return out
    

@torch.amp.autocast('cuda', enabled=False)    
def _lift_K(K):
    """
    [*, 3, 3] -> [*, 4, 4]
    """
    out = torch.zeros(
        *K.shape[:-2],
        4,
        4,
        device=K.device,
        dtype=K.dtype,
    )

    out[..., :3, :3] = K
    out[..., 3, 3] = 1.0

    return out


@torch.amp.autocast('cuda', enabled=False)
def _invert_se3(T):
    """
    T is world -> camera.

    [*, 4, 4]
    """
    R_t = T[..., :3, :3].transpose(-1, -2)

    out = torch.zeros_like(T)

    out[..., :3, :3] = R_t
    out[..., :3, 3] = -torch.einsum(
        "...ij,...j->...i",
        R_t,
        T[..., :3, 3],
    )
    out[..., 3, 3] = 1.0

    return out


@torch.amp.autocast('cuda', enabled=False)
def _apply_projective(
    x: torch.Tensor,
    matrix: torch.Tensor
):
    """
    Apply a per-token 4x4 matrix over the 1/2 head dimension
    
    Args
    x: [S, N, D]
    matrix: [valid_len, 4, 4]
    """
    
    S, N, D = x.shape
    
    assert D % 4 ==0
    valid_len = matrix.shape[0]
    
    dtype = x.dtype
    
    x_4 = x.float().reshape(S, N, D//4, 4)
    x_valid = x_4[:valid_len]
    matrix = matrix.float()
    
    x_valid = torch.einsum(
        "sij,snkj->snki",
        matrix,
        x_valid,
    )
    
    x_4 = torch.cat([x_valid, x_4[valid_len:]], dim=0)
   
    return x_4.reshape(S, N, D).to(dtype)


@torch.amp.autocast('cuda', enabled=False)
def prope_apply(q, k, v, mk, mv, K, viewmats, memory_viewmats, seq_len, mem_len):
    """
    Args:
    x [B, S, N, D] (q, k, v, mk, mv)
    K [B, 3, 3] normalized
    viewmats: [B, S, 4, 4] world_to_camera
    memory_viewmats [B, max_mem_len, 4, 4]
    seq_len [B]
    mem_len [B]
    """
    B, S, N, D = q.shape
    
    K4 = _lift_K(K)
    
    K_inv = torch.linalg.inv(K)
    K4_inv = _lift_K(K_inv)
    
    P_seq = [K4[i][None] @ viewmats[i][:seq_len[i]] for i in range(B)]
    P_inv_seq = [_invert_se3(viewmats[i][:seq_len[i]]) @ K4_inv[i][None] for i in range(B)]

    
    P_inv_mem = [_invert_se3(memory_viewmats[i][:mem_len[i]]) @ K4_inv[i][None] for i in range(B)]
    
    prope_q = torch.stack([_apply_projective(q[i], P_seq[i].transpose(-1, -2)) for i in range(B)])
    prope_k = torch.stack([_apply_projective(k[i], P_inv_seq[i]) for i in range(B)])
    prope_v = torch.stack([_apply_projective(v[i], P_inv_seq[i]) for i in range(B)])
    
    
    prope_mk = torch.stack([_apply_projective(mk[i], P_inv_mem[i]) for i in range(B)])
    prope_mv = torch.stack([_apply_projective(mv[i], P_inv_mem[i]) for i in range(B)])
    
    return prope_q, prope_k, prope_v, prope_mk, prope_mv, P_seq 


@torch.amp.autocast('cuda', enabled=False)
def prope_rope_apply(q, k, v, mk, mv, grid_sizes, freqs, grid_fuv, K, viewmats, memory_viewmats):
    """
    Args:
    q,k,v: [B, S, N, D]
    mk,mv: [B, max_mem_len, N, D] qkv,mk,mv都是padding后的,所以后面cat的时候注意去掉mk,mv的padding再与kvcat
    grid_sizes: [B,3]
    freqs: [1024, dim/2]
    memory_len:[B]
    grid_fuv:List [Tensor] each with shape [memory_len, 3]
    K:List [Tensor] each with shape [3,3]
    viewmats: List [Tensor] each with shape [F, 4 , 4] 
    memory_viewmats: List [Tensor] each with shape [memory_len, 4, 4]
    """
    # 首先进行RoPE的编码
    rope_q = qkv_rope_apply(q, grid_sizes, freqs)
    rope_k = qkv_rope_apply(k, grid_sizes, freqs)
    
    rope_mk = mk_rope_apply(mk, grid_fuv, freqs)
    
    # 然后进行PRoPE的编码
    
    # 对K进行normalize
    h = grid_sizes[:, 1] * 32
    w = grid_sizes[:, 2] * 32
    K = _normalize_K(K, h, w)
    
    seq_len = [grid_sizes[i][0] * grid_sizes[i][1] * grid_sizes[i][2] for i in range(grid_sizes.size(0))]
    max_seq_len = max(seq_len).item()
    mem_len = [fuv.size(0) for fuv in grid_fuv]
    max_mem_len = max(mem_len)
    
    # 对viewmats进行处理 to [B, S, 4, 4] 
    tokens_per_frame = [grid_sizes[i][1] * grid_sizes[i][2] for i in range(grid_sizes.size(0))]
    viewmats = [viewmats[i].repeat_interleave(tokens_per_frame[i], 0) for i in range(len(viewmats))]
    viewmats = torch.stack([torch.cat([viewmats[i], viewmats[i].new_zeros(max_seq_len - seq_len[i], 4, 4)], dim=0) for i in range(len(seq_len))])
    memory_viewmats = torch.stack([torch.cat([memory_viewmats[i], memory_viewmats[i].new_zeros(max_mem_len - mem_len[i], 4, 4)], dim=0) for i in range(len(mem_len))])
    prope_q, prope_k, prope_v, prope_mk, prope_mv, P_seq = prope_apply(q, k, v, mk, mv, K, viewmats, memory_viewmats, seq_len, mem_len)
    
    # 在序列维拼接mk,mv
    rope_k = torch.stack([torch.cat([rope_mk[i][:mem_len[i]], rope_k[i][:seq_len[i]], rope_k[i].new_zeros(max_seq_len + max_mem_len - seq_len[i] - mem_len[i], k[i].size(1), k[i].size(2))], dim=0) for i in range(mk.size(0))])
    prope_k = torch.stack([torch.cat([prope_mk[i][:mem_len[i]], prope_k[i][:seq_len[i]], prope_k[i].new_zeros(max_seq_len + max_mem_len - seq_len[i] - mem_len[i], k[i].size(1), k[i].size(2))], dim=0) for i in range(mk.size(0))])
    
    prope_v = torch.stack([torch.cat([prope_mv[i][:mem_len[i]], prope_v[i][:seq_len[i]], prope_v[i].new_zeros(max_seq_len + max_mem_len - seq_len[i] - mem_len[i], k[i].size(1), k[i].size(2))], dim=0) for i in range(mk.size(0))])
    
    v = torch.stack([torch.cat([mv[i][:mem_len[i]], v[i][:seq_len[i]], v[i].new_zeros(max_seq_len + max_mem_len - seq_len[i] - mem_len[i], k[i].size(1), k[i].size(2))], dim=0) for i in range(mk.size(0))])
    
    # 在batch维度拼接
    q = torch.cat([rope_q, prope_q], dim=0)
    k = torch.cat([rope_k, prope_k], dim=0)
    v = torch.cat([v, prope_v], dim=0)
    
    return q, k, v, P_seq

class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)
       
# 仿照WorldPlay的方式增加PRoPE branch
class WanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, K, viewmats, memory_tokens, memory_len, memory_viewmats, grid_fuv):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        ms = memory_len.max().item()

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        
        def kv_mem_fn(x):
            mk = self.norm_k(self.k(x)).view(b, ms, n, d)
            mv = self.v(x).view(b, ms, n, d)
            return mk, mv
        
        mk, mv = kv_mem_fn(memory_tokens)
    
        q, k, v, P = prope_rope_apply(q, k, v, mk, mv, grid_sizes, freqs, grid_fuv, K, viewmats, memory_viewmats)
        valid_len = seq_lens + memory_len
        valid_len = torch.cat([valid_len, valid_len], dim=0)
        x = flash_attention(
            q,
            k,
            v=v,
            k_lens=valid_len,
            window_size=self.window_size)
        
        # 在batch维度分开 rope 和 prope
        x_rope, x_prope = torch.chunk(x, 2, dim=0)
        # 对x_prope应用 _apply_projective
        x_prope = torch.stack([_apply_projective(x_prope[i], P[i]) for i in range(b)])
        x = x_rope + x_prope

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanCrossAttention(WanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        x = flash_attention(q, k, v, k_lens=context_lens)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanAttentionBlock(nn.Module):

    def __init__(self,
                 dim,
                 ffn_dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=False,
                 eps=1e-6):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm,
                                          eps)
        self.norm3 = WanLayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WanCrossAttention(dim, num_heads, (-1, -1), qk_norm,
                                            eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim), nn.GELU(approximate='tanh'),
            nn.Linear(ffn_dim, dim))

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        K,
        viewmats,
        memory_tokens,
        memory_len,
        memory_viewmats,
        grid_fuv,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, L1, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            memory_len(List[int]): len of memory_tokens each with shape [M, C]
            viewmats(List[Tensor]) : each with shape [M, 4, 4]
            grid_uv(List[Tensor]) each with shape [M, 2]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e).chunk(6, dim=2)
        assert e[0].dtype == torch.float32
        
        attn_input = self.norm1(x).float() * (1 + e[1].squeeze(2)) + e[0].squeeze(2)

        # self-attention 增加PRoPE branch,通过增加一个batch来实现
        y = self.self_attn(
            attn_input,
            seq_lens, grid_sizes, freqs, K, viewmats, memory_tokens, memory_len, memory_viewmats, grid_fuv)
        
        x = x + y * e[2].squeeze(2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e):
            x = x + self.cross_attn(self.norm3(x), context, context_lens)
            y = self.ffn(
                self.norm2(x).float() * (1 + e[4].squeeze(2)) + e[3].squeeze(2))
            with torch.amp.autocast('cuda', dtype=torch.float32):
                x = x + y * e[5].squeeze(2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e)
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, L1, C]
        """
        assert e.dtype == torch.float32
        with torch.amp.autocast('cuda', dtype=torch.float32):
            e = (self.modulation.unsqueeze(0) + e.unsqueeze(2)).chunk(2, dim=2)
            x = (
                self.head(
                    self.norm(x) * (1 + e[1].squeeze(2)) + e[0].squeeze(2)))
        return x


class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

    ignore_for_config = [
        'patch_size', 'cross_attn_norm', 'qk_norm', 'text_dim', 'window_size'
    ]
    _no_split_modules = ['WanAttentionBlock']

    @register_to_config
    def __init__(self,
                 model_type='t2v',
                 patch_size=(1, 2, 2),
                 text_len=512,
                 in_dim=16,
                 dim=2048,
                 ffn_dim=8192,
                 freq_dim=256,
                 text_dim=4096,
                 out_dim=16,
                 num_heads=16,
                 num_layers=32,
                 window_size=(-1, -1),
                 qk_norm=True,
                 cross_attn_norm=True,
                 eps=1e-6):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ['t2v', 'i2v', 'ti2v', 's2v']
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList([
            WanAttentionBlock(dim, ffn_dim, num_heads, window_size, qk_norm,
                              cross_attn_norm, eps) for _ in range(num_layers)
        ])

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads
        self.freqs = torch.cat([
            rope_params(1024, d - 4 * (d // 6)),
            rope_params(1024, 2 * (d // 6)),
            rope_params(1024, 2 * (d // 6))
        ],
                               dim=1)

        # initialize weights
        self.init_weights()

    def forward(
        self,
        x,
        K,
        viewmats,
        memory_tokens, 
        memory_viewmats,
        grid_fuv, 
        t,
        context,
        seq_len,
        y=None,
        training=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            viewmats:
            memory_tokens (List[Tensor]):
                List of input memory tokens tensors, each with shape [N, C] N: num, C: channel_dim
            memory_viewmats: 大小是 max_memory_size(用padding补齐)
            grid_uv (List[Tensor]):
                List of grid_uv of memory tokens, each with shape [N, 2]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if self.model_type == 'i2v':
            assert y is not None
        # params
        device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]
            
        assert len(x) == len(memory_tokens)
    
        # 得到每个batch的memory_len
        memory_len = torch.tensor([tokens.size(0) for tokens in memory_tokens])
        max_mem_len = memory_len.max().item()
        memory_tokens= torch.stack([
            torch.cat([tokens, tokens.new_zeros(max_mem_len - tokens.size(0), tokens.size(1))], dim=0) for tokens in memory_tokens
        ])
    
        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x] # 其中的u变成了 [1, C, T, H/2, W/2],即每个token的位置
        grid_sizes = torch.stack(
            [torch.tensor(u.shape[2:], dtype=torch.long, device=u.device) for u in x])
        x = [u.flatten(2).transpose(1, 2) for u in x] # [B, 1, N, C]
        
        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        assert seq_lens.max() <= seq_len
        x = torch.cat([
            torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))],
                      dim=1) for u in x
        ])
        
        # time embeddings
        if t.dim() == 1:
            t = t.expand(t.size(0), seq_len)
        with torch.amp.autocast('cuda', dtype=torch.float32):
            bt = t.size(0)
            t = t.flatten()
            e = self.time_embedding(
                sinusoidal_embedding_1d(self.freq_dim,
                                        t).unflatten(0, (bt, seq_len)).float())
            e0 = self.time_projection(e).unflatten(2, (6, self.dim))
            assert e.dtype == torch.float32 and e0.dtype == torch.float32

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([
                torch.cat(
                    [u, u.new_zeros(self.text_len - u.size(0), u.size(1))])
                for u in context
            ]))

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            K=K,
            viewmats=viewmats,
            memory_tokens = memory_tokens,
            memory_len=memory_len,
            memory_viewmats=memory_viewmats,
            grid_fuv=grid_fuv)

        for block in self.blocks:
            if self.training:
                x = checkpoint.checkpoint(
                    block,
                    x,
                    **kwargs,
                    use_reentrant=False,
                )
            else:
                x = block(x, **kwargs)
        
        # head
        x = self.head(x, e)

        # unpatchify
        x = self.unpatchify(x, grid_sizes)
        return [u.float() for u in x]

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            u = u[:math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum('fhwpqrc->cfphqwr', u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
        
    
    

