import torch

class LinearWarmupDecay:
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float,
        warmup_steps: int,
        decay_start_step: int,
        decay_steps: int,
        min_lr: float,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.decay_start_step = decay_start_step
        self.decay_steps = decay_steps
        self.min_lr = min_lr
        self.step_num = 0
        self._set_lr(0.0)

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self, step: int) -> float:
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return self.base_lr * (step / self.warmup_steps)
        if step < self.decay_start_step:
            return self.base_lr
        if step < self.decay_start_step + self.decay_steps:
            progress = (step - self.decay_start_step) / self.decay_steps
            return self.base_lr + (self.min_lr - self.base_lr) * progress
        return self.min_lr

    def step(self) -> float:
        self.step_num += 1
        lr = self.get_lr(self.step_num)
        self._set_lr(lr)
        return lr
