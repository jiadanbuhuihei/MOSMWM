import abc
import logging
import os
import sys
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from ..utils.config import instantiate_from_config
from ..utils.checkpoint import CheckpointManager
from ..utils.init_distributed import init_distributed
from ..models.wan_vae import Wan2_2_VAE
from ..models.t5 import T5EncoderModel
from ..models.wan22 import WanModel
from ..data.dataloader import build_data_loader

def setup_logging(rank):

    logging.basicConfig(
        level=logging.INFO,
        format=f"[rank {rank}] [%(asctime)s][%(name)s][%(levelname)s] - %(message)s",
        handlers=[
            logging.FileHandler(
                f"world_model/train_rank{rank}.log",
                mode="w"
            ),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )

logger = logging.getLogger(__name__)

# 定义基本的调用接口，然后不同阶段的训练器有各自的实现
# 基础工作： 1. 加载数据集（训练和评估） 2. 加载模型 3. 给定统一调用接口

class BaseTrainer(abc.ABC):
    
    def __init__(
        self,
        base_seed,
        default_config, # 默认配置文件
        dataset_config, # 数据集相关参数
        optimizer_config, # 优化器相关参数
        scheduler_config, # 学习率调度器相关参数
        training_config, # 训练相关参数
        checkpoint_dir, # 预训练模型所在的文件夹地址
        cfg, # 整个训练的所有参数，用于checkpoint保存
    ):
        self.base_seed = base_seed
        self.default_config = default_config
        self.dataset_config = dataset_config
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        self.training_config = training_config
        self.checkpoint_dir = checkpoint_dir
        self.cfg = cfg
        
        self.local_rank, self.rank, self.world_size = init_distributed()
        
        self.device = torch.device(f"cuda:{self.local_rank}")
        
        setup_logging(self.rank)
        
        # 加载VAE,t5 (封装类已经冻结了)
        self.vae = Wan2_2_VAE(
            vae_pth=os.path.join(self.checkpoint_dir, self.default_config.vae_checkpoint),
            device = self.device
        )
        
        self.text_encoder = T5EncoderModel(
            text_len=512,
            dtype=torch.bfloat16,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(self.checkpoint_dir, self.default_config.t5_checkpoint),
            tokenizer_path=os.path.join(self.checkpoint_dir, self.default_config.t5_tokenizer),
        )
        
        self.neg_prompt = self.default_config.sample_neg_prompt
   
    @abc.abstractmethod
    def get_cur_batch(self, train_loader_iter):
        pass
    
    @abc.abstractmethod
    def _train_step(self):
        pass
    
    @abc.abstractmethod
    def _test_step(self):
        pass
    
    def load_train_and_eval_datasets(self):
        self.train_dataset = instantiate_from_config(self.dataset_config["train_dataset"])
        self.test_dataset = instantiate_from_config(self.dataset_config["test_dataset"])
        
    def load_pretrained_models(self):
        self.model = WanModel.from_pretrained(self.default_config["pretrained_wan22_path"],torch_dtype=torch.bfloat16)
        self.model.to(self.device)

        if self.world_size > 1:
            self.model = DDP(
                self.model,
                device_ids=[self.local_rank],
                output_device=self.local_rank,
                find_unused_parameters=False,
            ) 
    
    def build_optimizer(self):
        self.optimizer = instantiate_from_config(self.optimizer_config, params=self.model.parameters())

    def build_scheduler(self):
        self.scheduler = instantiate_from_config(self.scheduler_config, optimizer=self.optimizer)
        

    # 统一调用接口
    def run(self, training_signal):
        # 1 加载数据集
        logger.info("Loading train and eval datasets...")
        self.load_train_and_eval_datasets()
        self.train_loader = build_data_loader(
            self.train_dataset,
            self.default_config["batch_size"],
            self.default_config["num_workers"],
            self.default_config["num_frames"],
            seed_data=self.base_seed,
            eval=False,
            rank=self.rank,
            world_size=self.world_size,
        )
       
        self.test_loader = build_data_loader(
            self.test_dataset,
            self.default_config["batch_size"],
            self.default_config["num_workers"],
            self.default_config["num_frames"],
            seed_data=self.base_seed,
            eval=False,
            rank=self.rank,
            world_size=self.world_size,
        )   
        
        # 2. 加载预训练模型
        logger.info("Loading pretrained models...")
        self.load_pretrained_models()           
        
        # 3. 构建优化器
        logger.info("Building optimizer...")
        self.build_optimizer()
        
        # 4. 构建scheduler
        logger.info("Building Scheduler...")
        self.build_scheduler()
        
        # 4. 创建CheckpointManager
        logger.info("Creating CheckpointManager...")
        restore_dir = os.path.join(self.checkpoint_dir, self.training_config["restore_dir"])
        os.makedirs(restore_dir, exist_ok=True)
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=restore_dir)
        
        step = 0  # Initialize step to -1 to indicate no training has occurred yet
        
        if self.training_config["restore"]:
            logger.info("Restoring from checkpoint...")
            assert self.training_config["restore_path"] is not None, "restore_path must be provided when restore is True"
            step = self.checkpoint_manager.load(self.training_config["restore_path"], self.model, self.optimizer, self.scheduler, self.train_loader)
        
        
        train_loader_iter = iter(self.train_loader)
        test_loader_iter = iter(self.test_loader)
        
        # 创建训练循环
        logger.info("Starting training loop...")
        self.model.train()
        num_steps = self.training_config["num_steps"]
        train_log_every_steps = self.training_config.train_log_every_steps
        test_every_steps = self.training_config.test_every_steps
        test_num_steps = self.training_config.test_num_steps
        save_every_steps = self.training_config.save_every_steps    

        total_loss = []
        total_grad_norm = []
        for i in range(step + 1, num_steps + 1):
            
            # 获得本轮的batch(dict)
            batch = self.get_cur_batch(train_loader_iter)
            
            self.optimizer.zero_grad()
            
            loss = self._train_step(batch)
            
            loss.backward()
            
            total_loss.append(loss.item())
            grad_norm = clip_grad_norm_(
                            self.model.parameters(),
                            float("inf")
                        ).item()
            total_grad_norm.append(grad_norm)

            self.optimizer.step()

            self.scheduler.step()

            if i % train_log_every_steps == 0 or i == num_steps:
                # 每 train_log_every_steps 步后记录一次loss和grad_norm,param_norm
                logger.info(f"Train_log step {i} ...")
                L = len(total_loss)
                mean_loss = sum(total_loss) / L
                mean_grad_norm  = sum(total_grad_norm) / L
                param_norm  = self.get_param_norm(self.model)
                
                total_loss.clear()
                total_grad_norm.clear()
                logger.info(f"Step {i}: mean_loss: {mean_loss}")
                logger.info(f"Step {i}: mean_grad_norm: {mean_grad_norm}")
                logger.info(f"Step {i}: param_norm: {param_norm}")
                
            if i % test_every_steps == 0 or i == num_steps:
                # 每 test_every_steps 步后在 testdata 上进行验证
                logger.info(f"Test in step {i} ...")
                self.model.eval()
                
                test_loss = 0.0
                for _ in range(test_num_steps):
                    batch = self.get_cur_batch(test_loader_iter)
                    test_loss += self._test_step(batch)
                test_loss /= test_num_steps
                logger.info(f"Test mean_loss: {test_loss}")
                
                self.model.train()
                
            if (i % save_every_steps == 0 or i == num_steps) and self.rank == 0:
                logger.info(f"Saving model state and training state in Step {i}")
                self.checkpoint_manager.save(self.model, self.optimizer, self.scheduler, self.train_loader, i, self.cfg)
            
            if training_signal.stop_training:
                if self.rank == 0:
                    logger.info("Training stopped by signal, saveing model state...")
                    self.checkpoint_manager.save(self.model, self.optimizer, self.scheduler, self.train_loader, i, self.cfg)
                    break
        
        
    def robust_batch_sample(self, loader_iter, num_retries=5):

        for _ in range(num_retries):
            try:
                return next(loader_iter)
            except Exception:
                logging.info(f"rank:{self.rank} retrying to load batch")
        raise RuntimeError("Failed to load batch")
                
    def get_param_norm(self, model):
        total_norm = 0.0

        for p in model.parameters():
            if p.requires_grad:
                param_norm = p.detach().float().norm(2)
                total_norm += param_norm.item() ** 2

        return total_norm ** 0.5