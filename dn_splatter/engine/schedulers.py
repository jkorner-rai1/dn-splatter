from dataclasses import dataclass, field
from typing import Type, Optional
import numpy as np
from torch.optim import lr_scheduler, Optimizer
from nerfstudio.configs.base_config import InstantiateConfig
from nerfstudio.engine.schedulers import ExponentialDecaySchedulerConfig, ExponentialDecayScheduler
from torch.optim import lr_scheduler, Optimizer

from dataclasses import dataclass, field
from typing import Type, Optional


@dataclass
class FreezeThenFixedSchedulerConfig(ExponentialDecaySchedulerConfig):
	"""Scheduler config: lr=0 for freeze_steps, then fixed lr."""
	_target: Type = field(default_factory=lambda: FreezeThenFixedScheduler)
	freeze_steps: int = 1000
	fixed_lr: Optional[float] = None  # If None, use lr_init
	"""Number of steps to freeze lr (set to 0), then fixed lr."""

class FreezeThenFixedScheduler(ExponentialDecayScheduler):
	config: FreezeThenFixedSchedulerConfig

	def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler.LambdaLR:
		fixed_lr = self.config.fixed_lr if self.config.fixed_lr is not None else lr_init

		def func(step):
			if step < self.config.freeze_steps:
				return 0.0
			else:
				return fixed_lr / lr_init

		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
		return scheduler


# --- New Scheduler: FixedThenOneScheduler ---
@dataclass
class FixedThenOneSchedulerConfig(ExponentialDecaySchedulerConfig):
	"""Scheduler config: fixed lr for warmup_steps, then 1.0 (no scaling)."""
	_target: Type = field(default_factory=lambda: FixedThenOneScheduler)
	warmup_steps: int = 1000
	fixed_lr: Optional[float] = None  # If None, use lr_init
	"""Number of steps to use fixed lr, then 1.0."""

class FixedThenOneScheduler(ExponentialDecayScheduler):
	config: FixedThenOneSchedulerConfig

	def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler.LambdaLR:
		fixed_lr = self.config.fixed_lr if self.config.fixed_lr is not None else lr_init

		def func(step):
			if step < self.config.warmup_steps:
				return fixed_lr / lr_init
			else:
				return 1.0

		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
		return scheduler


# --- New Scheduler: MultiStepExplicitLRScheduler ---
@dataclass
class MultiStepLRSchedulerConfig(ExponentialDecaySchedulerConfig):
	"""Scheduler config: set explicit learning rates at given steps."""
	_target: Type = field(default_factory=lambda: MultiStepLRScheduler)
	milestones: list = field(default_factory=list)  # List of step indices
	lrs: list = field(default_factory=list)         # List of learning rates (same length as milestones+1)
	"""milestones: steps at which to change lr; lrs: explicit lr values for each interval."""

class MultiStepLRScheduler(ExponentialDecayScheduler):
	config: MultiStepLRSchedulerConfig

	def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler.LambdaLR:
		milestones = self.config.milestones
		lrs = self.config.lrs
		if not lrs:
			lrs = [lr_init]
		def func(step):
			for i, m in enumerate(milestones):
				if step < m:
					return lrs[i] / lr_init
			return lrs[-1] / lr_init
		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
		return scheduler


@dataclass
class FreezeThenDecaySchedulerConfig(ExponentialDecaySchedulerConfig):
	"""Scheduler config: lr=0 for freeze_steps, then exponential decay."""
	_target: Type = field(default_factory=lambda: FreezeThenDecayScheduler)
	"""Number of steps to freeze lr (set to 0)."""
	freeze_steps: int = 1000
	

class FreezeThenDecayScheduler(ExponentialDecayScheduler):
	config: FreezeThenDecaySchedulerConfig

	def get_scheduler(self, optimizer: Optimizer, lr_init: float) -> lr_scheduler.LambdaLR:
		if self.config.lr_final is None:
			lr_final = lr_init
		else:
			lr_final = self.config.lr_final

		def func(step):
			if step < self.config.freeze_steps:
				return 0.0
			# After freeze, use normal exponential decay with warmup
			step_adj = step - self.config.freeze_steps
			warmup_steps = max(self.config.warmup_steps - self.config.freeze_steps, 0)
			if step_adj < warmup_steps:
				if self.config.ramp == "cosine":
					lr = self.config.lr_pre_warmup + (lr_init - self.config.lr_pre_warmup) * np.sin(
						0.5 * np.pi * np.clip(step_adj / warmup_steps, 0, 1)
					)
				else:
					lr = (
						self.config.lr_pre_warmup
						+ (lr_init - self.config.lr_pre_warmup) * step_adj / warmup_steps
					)
			else:
				t = np.clip(
					(step_adj - warmup_steps) / (self.config.max_steps - warmup_steps), 0, 1
				)
				lr = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
			return lr / lr_init

		scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=func)
		return scheduler
