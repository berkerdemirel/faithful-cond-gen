import torch
import torch.nn as nn


class EMA:
    def __init__(self, module: torch.nn.Module, decay: float = 0.9999):
        self.module = module
        self.decay = decay
        self.shadow = {}
        self._backup = {}

        # track only trainable params
        for name, p in module.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.detach().float().clone()

    @torch.no_grad()
    def update(self):
        d = self.decay
        for name, p in self.module.named_parameters():
            if not p.requires_grad:
                continue
            self.shadow[name].mul_(d).add_(p.detach().float(), alpha=1.0 - d)

    @torch.no_grad()
    def apply(self):
        self._backup = {}
        for name, p in self.module.named_parameters():
            if not p.requires_grad:
                continue
            self._backup[name] = p.detach().clone()
            p.data.copy_(self.shadow[name].to(p.dtype).to(p.device))

    @torch.no_grad()
    def restore(self):
        for name, p in self.module.named_parameters():
            if not p.requires_grad:
                continue
            p.data.copy_(self._backup[name])
        self._backup = {}

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, sd):
        self.decay = sd["decay"]
        self.shadow = sd["shadow"]
