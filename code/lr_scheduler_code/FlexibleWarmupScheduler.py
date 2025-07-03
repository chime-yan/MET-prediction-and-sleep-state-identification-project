class FlexibleWarmupScheduler:
    def __init__(self, optimizer,
                 warmup_steps=0,
                 warmup_epochs=0,
                 min_lr=1e-7,
                 max_lr=1e-3):
        """
        通用 Warmup 调度器

        参数：
        - optimizer: 优化器
        - warmup_steps: 使用 step 级 warmup，总共多少 step warmup，0 表示不使用 step warmup
        - warmup_epochs: 使用 epoch 级 warmup，总共多少 epoch warmup，0 表示不使用 epoch warmup
        - min_lr: warmup 起始学习率
        - max_lr: warmup 最终学习率
        """

        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.max_lr = max_lr

        self.step_num = 0
        self.epoch_num = 0
        self.finished = (warmup_steps == 0 and warmup_epochs == 0)

        # 初始化 lr 为 min_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr

    def _set_lr(self, lr):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

    def _get_lr(self, current, total):
        ratio = min(float(current) / total, 1.0)
        return self.min_lr + ratio * (self.max_lr - self.min_lr)

    def step_batch(self):
        if self.warmup_steps > 0 and not self.finished:
            self.step_num += 1
            lr = self._get_lr(self.step_num, self.warmup_steps)
            self._set_lr(lr)
            if self.step_num >= self.warmup_steps:
                self.finished = True

    def step_epoch(self):
        if self.warmup_epochs > 0 and not self.finished:
            self.epoch_num += 1
            lr = self._get_lr(self.epoch_num + 1, self.warmup_epochs)
            self._set_lr(lr)
            if self.epoch_num >= self.warmup_epochs:
                self.finished = True

    def is_finished(self):
        return self.finished


# 使用示例
# warmup_scheduler = FlexibleWarmupScheduler(
#     optimizer=optimizer,
#     warmup_epochs=0,
#     max_lr=1e-3,
#     min_lr=1e-7
# )
