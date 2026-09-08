import math
from torch.optim.lr_scheduler import LambdaLR

class CosineWarmupScheduler(LambdaLR):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        min_lr_ratio=0.0,
    ):

        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio

        super().__init__(
            optimizer,
            self.lr_lambda
        )


    def lr_lambda(self, step):

        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)

        progress = (
            step - self.warmup_steps
        ) / (
            self.total_steps - self.warmup_steps
        )

        cosine = 0.5 * (
            1 + math.cos(math.pi * progress)
        )

        return (
            self.min_lr_ratio
            +
            (1 - self.min_lr_ratio) * cosine
        )