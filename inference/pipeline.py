import logging
import os
import gc
import random
import math
import sys
from contextlib import contextmanager


import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from ..models.wan_vae import Wan2_2_VAE
from ..models.wan22 import WanModel
from ..models.t5 import T5EncoderModel
from ..models.utils import best_output_size, masks_like, _invert_se3
from ..utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from ..utils.fm_solvers_unipc import FlowUniPCMultistepScheduler

class SPModel:
    
    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id,
        t5_cpu,
    ):
        self.device = torch.device(f"cuda:{device_id}")
        self.t5_cpu = t5_cpu
        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype
        
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device('cpu'),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device
        )
        
        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        
        self.sample_neg_prompt = config.sample_neg_prompt
        
    def __call__(
        self,
        K,
        viewmats,
        memory_tokens,
        memory_viewmats,
        grid_fuv,
        img,
        max_area=704 * 1280,
        frame_num=81,
        shift=5.0,
        sample_solver = 'unipc',
        sampling_steps=50,
        guide_scale=5.0,
        input_prompt="",
        n_prompt="",
        seed=-1,
        offload_model=True
    ):
        """
        K: 相机内参 [B, 3, 3]
        viewmats: 相机位姿 [B, T, 4, 4]
        memory_tokens: 检索得到的记忆tokens List[] each with shape [mem_len, C] 这里的C是3072
        memory_viewmats: 每个memory_tokens对应的相机位姿
        grid_fuv: 每个memory_tokens对应的fuv值,用于编码RoPE
        img: I2V输入的image
        max_area: 限制的生成视频的最大分辨率面积,用于求最合适的h,w
        frame_num: 最后生成的视频的帧数(包含img作为第一帧)
        shift: Noise schedule shift parameter
        samole_solver: Solver used to sample the video
        sampling_steps: Number of diffusion sampling steps
        guide_scale: CFG scale
        n_prompt: Negative prompt for content exclusion
        seed: Random seed for noise generation
        offload_model: If True, offloads models to CPU during generation to save VRAM
        """
        # preprocess
        ih, iw = img.height, img.width
        dh, dw = self.patch_size[1] * self.vae_stride[1], self.patch_size[
            2] * self.vae_stride[2]
        ow, oh = best_output_size(iw, ih, dw, dh, max_area)

        scale = max(ow / iw, oh / ih)
        img = img.resize((round(iw * scale), round(ih * scale)), Image.LANCZOS)

        # center-crop
        x1 = (img.width - ow) // 2
        y1 = (img.height - oh) // 2
        img = img.crop((x1, y1, x1 + ow, y1 + oh))
        assert img.width == ow and img.height == oh

        # to tensor
        img = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device).unsqueeze(1)

        F = frame_num
        # 因为添加了memory tokens，而memory tokens有上限，上限为8帧的tokens (当然可以调整，这里为了方便且一般规定后不会再改)
        # 所以这里在 帧的维度添加8
        seq_len = ((F - 1) // self.vae_stride[0] + 1) * (
            oh // self.vae_stride[1]) * (ow // self.vae_stride[2]) // (
                self.patch_size[1] * self.patch_size[2])

        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)
        noise = torch.randn(
            self.vae.model.z_dim, (F - 1) // self.vae_stride[0] + 1,
            oh // self.vae_stride[1],
            ow // self.vae_stride[2],
            dtype=torch.float32,
            generator=seed_g,
            device=self.device)

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt

        # preprocess
        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device('cpu'))
            context_null = self.text_encoder([n_prompt], torch.device('cpu'))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        z = self.vae.encode([img])

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, 'no_sync', noop_no_sync)

        # evaluation mode
        with (
                torch.amp.autocast('cuda', dtype=self.param_dtype),
                torch.no_grad(),
                no_sync(),
        ):

            if sample_solver == 'unipc':
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift)
                timesteps = sample_scheduler.timesteps
            elif sample_solver == 'dpm++':
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False)
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latent = noise
            mask1, mask2 = masks_like([noise], zero=True)
            latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

            arg_c = {
                'context': [context[0]],
                'seq_len': seq_len,
            }

            arg_null = {
                'context': context_null,
                'seq_len': seq_len,
            }

            if offload_model:
                self.model.to(self.device)
                torch.cuda.empty_cache()
            
            viewmats = _invert_se3(viewmats)
            memory_viewmats = _invert_se3(memory_viewmats)
            
            K = K.unsqueeze(0).to(self.device)
            viewmats = [viewmats.to(self.device)]
            memory_tokens = [memory_tokens.to(self.device)] # m因为每个batch的memory-token数量并不一致，所以这里统一用List作为接口
            memory_viewmats = [memory_viewmats.to(self.device)]
            grid_fuv = [grid_fuv.to(self.device)]

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = [latent.to(self.device)]
                timestep = [t]

                timestep = torch.stack(timestep).to(self.device)

                temp_ts = (mask2[0][0][:, ::2, ::2] * timestep).flatten() # 这里把latent mask 转换成Transformer patch mask
                temp_ts = torch.cat([
                    temp_ts,
                    temp_ts.new_ones(seq_len - temp_ts.size(0)) * timestep
                ]) # 前面为了self.sp_size整数倍补齐过seq_len，所有这里也要补齐timestep
                timestep = temp_ts.unsqueeze(0) # 增加Batch维度，同一接口

                # 返回的是List[] 每个都是denoise后的latent,[0]之后就是latent
                noise_pred_cond = self.model(
                    latent_model_input, K, viewmats, memory_tokens, memory_viewmats, grid_fuv, t=timestep, **arg_c)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred_uncond = self.model(
                    latent_model_input, K, viewmats, memory_tokens, memory_viewmats, grid_fuv, t=timestep, **arg_null)[0]
                if offload_model:
                    torch.cuda.empty_cache()
                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond)

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latent.unsqueeze(0),
                    return_dict=False,
                    generator=seed_g)[0]
                latent = temp_x0.squeeze(0)
                latent = (1. - mask2[0]) * z[0] + mask2[0] * latent

                x0 = [latent]
                del latent_model_input, timestep
                
            # 拿最后得到的latent再过一遍 Wan 的 patch_embedding得到tokens, 但是我们要去掉第一帧latent，因为我们只取新生成的帧作为记忆
            latent = latent[:,1:]
            latent_input = latent.unsqueeze(0)
            latent_input = latent_input.to(
                device=self.model.patch_embedding.weight.device,
                dtype=self.model.patch_embedding.weight.dtype
            )
            memory = self.model.patch_embedding(latent_input)
            memory = memory.flatten(2).transpose(1,2) # [1, L, D]   

            if offload_model:
                self.model.cpu()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            videos = self.vae.decode(x0)

        del noise, x0
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
            
        return videos[0], memory[0]
        
        
        