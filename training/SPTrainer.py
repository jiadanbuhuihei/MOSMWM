
import torch

from .BaseTrainer import BaseTrainer


class SPTrainer(BaseTrainer):
    
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        pass
    
    def get_cur_batch(self, loader_iter):
        # 调用一次返回本轮的batch(解包后的，_train_step/test_step调用后直接获取扩散模型所需的输入参数)
        # 需要返回：K,viewmats,obs,act
        batch = self.robust_batch_sample(loader_iter)
        batch = batch.to_tensor(self.device)
        
        wan_batch = {
            "obs": [x for x in batch["obs"]],
            "act": [x for x in batch["act"]],
            "viewmats": [x for x in batch["viewmats"]],
            "K": batch["K"]
        }
        
        return wan_batch
    
    def get_latent(self, obs):
        # obs: List[] each with shape [F,H,W,C]
        obs_list = []
        for video in obs:
            video = video.permute(3,0,1,2)
            video = video.float()
            video  = video / 127.5 -1
            obs_list.append(video)
        with torch.no_grad():
            latent = self.vae.encode(obs_list)
        return latent
    
    def _train_step(self, batch):
        
        # if torch.rand(1) < 0.1:
        #     prompt = self.neg_prompt
        # else: 
        #     prompt = None 
        
        # 先把obs通过vae encode后得到target,并且构造latent
        with torch.no_grad():
            latent = self.get_latent(batch["obs"]) # List[] each with shape [C, F, H, W]
        
        # 保持第一帧干净，然后创建随机数来表示噪声强度，将噪声加到后面的帧上
        first_frame = [x[:,:1] for x in latent]
        future_latent = [x[:,1:] for x in latent]
        
        sigma = torch.rand(
            len(latent),
            device=self.device
        )
        t = (sigma * 1000).long()
        noise = [torch.randn_like(x) for x in future_latent]
        noisy_future = []
        
        for i, x in enumerate(future_latent):
            alpha = sigma[i]
            zt = (
                (1 - alpha) * x
                +
                alpha *  noise[i]
            )
            noisy_future.append(zt)
            
        model_input = []
        
        for first, noisy in zip(first_frame, noisy_future):
            x = torch.cat([first, noisy], dim=1)
            model_input.append(x)
            
        seq_len = (self.default_config["num_frames"] // 4 + 1) * 880
        
        B = len(model_input)
        
        with torch.no_grad():
            prompt = ["" for _ in range(B)]
            context = self.text_encoder(prompt, torch.device('cpu'))
            context = [u.to(self.device) for u in context]
            
        cond = {
            "x": model_input,
            "K": batch["K"].unsqueeze(0).repeat(B, 1, 1),
            "viewmats": batch["viewmats"],
            "t": t,
            "memory_tokens": [torch.empty((0, 3072),dtype=torch.float32, device=self.device) for _ in range(B)],
            "memory_viewmats": [torch.empty((0, 4, 4),dtype=torch.float32, device=self.device) for _ in range(B)],
            "grid_fuv": [torch.empty((0, 3), dtype=torch.long, device=self.device) for _ in range(B)],
            "context": context,
            "seq_len": seq_len,
        }

        with torch.amp.autocast(
            "cuda",
            dtype=torch.bfloat16
        ):
            pred = self.model(**cond)
        
        target = [n - x for n, x in zip(noise, future_latent)]
        
        mse_loss = 0.0
        
        for p, y in zip(pred, target):
            mse_loss += torch.nn.functional.mse_loss(p[:,1:,...].float(), y.float())
            
        mse_loss /= len(target)

        del pred, model_input, cond, noise, noisy_future, target
        torch.cuda.empty_cache()
        
        return mse_loss
    
    def _test_step(self, batch):
        with torch.no_grad():
            
            latent = self.get_latent(batch["obs"])
            first_frame = [x[:, :1] for x in latent]
            future_latent = [x[:, 1:] for x in latent]
            
            B = len(latent)
            
            sigma = torch.rand(B, device=self.device)
            
            t = (sigma * 1000).long()
            
            noise = [torch.randn_like(x) for x in future_latent]
            
            noisy_future = []
            
            for i, x in enumerate(future_latent):
                alpha = sigma[i]
                zt = (
                    (1 - alpha) * x
                    +
                    alpha * noise[i]
                )
                noisy_future.append(zt)
                
            model_input = []
            
            for first, noisy in zip(first_frame, noisy_future):
                x = torch.cat([first, noisy], dim=1)
                model_input.append(x)
                
            seq_len = (self.default_config["num_frames"] // 4 + 1) * 880
            
            prompt = ["" for _ in range(B)]
            context = self.text_encoder(prompt, torch.device('cpu'))
            context = [u.to(self.device) for u in context]
            
            cond = {
                "x": model_input,
                "K": batch["K"].unsqueeze(0).repeat(B, 1, 1),
                "viewmats": batch["viewmats"],
                "t": t,
                "memory_tokens": [torch.empty((0, 3072),dtype=torch.float32, device=self.device) for _ in range(B)],
                "memory_viewmats": [torch.empty((0, 4, 4),dtype=torch.float32, device=self.device) for _ in range(B)],
                "grid_fuv": [torch.empty((0, 3), dtype=torch.long, device=self.device) for _ in range(B)],
                "context": context,
                "seq_len": seq_len,
            }
            
            with torch.amp.autocast(
                "cuda",
                dtype=torch.bfloat16
            ):
                pred = self.model(**cond)
                
            target = [n - x for n, x in zip(noise, future_latent)]
            
            loss = 0.0
            
            for p, y in zip(pred, target):
                loss += torch.nn.functional.mse_loss(
                    p.float(),
                    y.float()
                )
                
            loss /= B
            
            return loss.item()
            