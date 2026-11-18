"""Mixture-of-LoRA experts for Qwen2-style linear layers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MoELoraConfig:
    num_experts: int = 4
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    )


def _adapter_device_dtype(base_linear: nn.Linear) -> tuple[torch.device, torch.dtype]:
    """Match new LoRA/router weights to the wrapped layer (4-bit base lives on GPU)."""
    device = base_linear.weight.device
    compute_dtype = getattr(base_linear, "compute_dtype", None)
    if compute_dtype is not None:
        return device, compute_dtype
    dtype = base_linear.weight.dtype
    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return device, dtype
    return device, torch.bfloat16


class LoRAExpert(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: int,
        dropout: float,
    ):
        super().__init__()
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_a = nn.Linear(in_features, rank, bias=False)
        self.lora_b = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=5**0.5)
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lora_b(self.dropout(self.lora_a(x))) * self.scaling


class MoELoraLinear(nn.Module):
    def __init__(
        self,
        base_linear: nn.Linear,
        num_experts: int,
        rank: int,
        alpha: int,
        dropout: float,
    ):
        super().__init__()
        self.base = base_linear
        self.base.weight.requires_grad_(False)
        if self.base.bias is not None:
            self.base.bias.requires_grad_(False)
        self.num_experts = num_experts
        device, dtype = _adapter_device_dtype(base_linear)
        self.router = nn.Linear(base_linear.in_features, num_experts, bias=False)
        self.experts = nn.ModuleList(
            [
                LoRAExpert(
                    base_linear.in_features,
                    base_linear.out_features,
                    rank,
                    alpha,
                    dropout,
                )
                for _ in range(num_experts)
            ]
        )
        self.router.to(device=device, dtype=dtype)
        for expert in self.experts:
            expert.to(device=device, dtype=dtype)
        self.adapter_dtype = dtype

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        x_a = x.to(dtype=self.adapter_dtype)
        gates = F.softmax(self.router(x_a), dim=-1)
        lora_out = torch.zeros_like(base_out)
        for i, expert in enumerate(self.experts):
            gate = gates[..., i : i + 1].to(dtype=base_out.dtype)
            expert_out = expert(x_a).to(dtype=base_out.dtype)
            lora_out = lora_out + gate * expert_out
        return base_out + lora_out

    def routing_weights(self, x: torch.Tensor) -> torch.Tensor:
        x_a = x if x.dtype == self.adapter_dtype else x.to(self.adapter_dtype)
        return F.softmax(self.router(x_a), dim=-1)


def _module_name(name: str, config: MoELoraConfig) -> bool:
    return any(name.endswith(suffix) or f".{suffix}" in name for suffix in config.target_modules)


def apply_moe_lora(model: nn.Module, config: MoELoraConfig) -> tuple[nn.Module, list[str]]:
    """Replace target linears with MoE-LoRA wrappers. Returns patched model and layer names."""
    patched: list[str] = []
    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not _module_name(name, config):
            continue
        parent_name, child_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        wrapped = MoELoraLinear(
            module,
            num_experts=config.num_experts,
            rank=config.rank,
            alpha=config.alpha,
            dropout=config.dropout,
        )
        setattr(parent, child_name, wrapped)
        patched.append(name)
    return model, patched


def trainable_moe_lora_params(model: nn.Module) -> list[nn.Parameter]:
    params: list[nn.Parameter] = []
    for module in model.modules():
        if isinstance(module, MoELoraLinear):
            params.extend(module.router.parameters())
            for expert in module.experts:
                params.extend(expert.parameters())
    return params


def _adapter_state_dict(module: MoELoraLinear) -> dict[str, torch.Tensor]:
    """Save only trainable MoE-LoRA weights (not the frozen 4-bit base layer)."""
    state: dict[str, torch.Tensor] = {}
    for key, value in module.router.state_dict().items():
        state[f"router.{key}"] = value.detach().cpu()
    for i, expert in enumerate(module.experts):
        for key, value in expert.state_dict().items():
            state[f"experts.{i}.{key}"] = value.detach().cpu()
    return state


def _load_adapter_state_dict(module: MoELoraLinear, state: dict[str, torch.Tensor]) -> None:
    """Load router/expert weights; ignore legacy base-layer keys from old checkpoints."""
    adapter_state = {k: v for k, v in state.items() if not k.startswith("base.")}
    router_state = {
        k.removeprefix("router."): v for k, v in adapter_state.items() if k.startswith("router.")
    }
    if router_state:
        module.router.load_state_dict(router_state)
    for i, expert in enumerate(module.experts):
        prefix = f"experts.{i}."
        expert_state = {k[len(prefix) :]: v for k, v in adapter_state.items() if k.startswith(prefix)}
        if expert_state:
            expert.load_state_dict(expert_state)


def save_moe_lora(model: nn.Module, out_dir: str, patched_layers: list[str], config: MoELoraConfig) -> None:
    import json
    from pathlib import Path

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    state = {
        name: _adapter_state_dict(model.get_submodule(name))
        for name in patched_layers
        if isinstance(model.get_submodule(name), MoELoraLinear)
    }
    torch.save(state, out / "moe_lora_adapters.pt")
    meta = {
        "adapter_type": "moe_lora",
        "patched_layers": patched_layers,
        "num_experts": config.num_experts,
        "rank": config.rank,
        "alpha": config.alpha,
        "dropout": config.dropout,
        "target_modules": list(config.target_modules),
    }
    with open(out / "moe_lora_config.json", "w") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")


def load_moe_lora(model: nn.Module, adapter_dir: str) -> nn.Module:
    import json
    from pathlib import Path

    adapter_path = Path(adapter_dir)
    with open(adapter_path / "moe_lora_config.json") as f:
        meta = json.load(f)
    config = MoELoraConfig(
        num_experts=meta["num_experts"],
        rank=meta["rank"],
        alpha=meta["alpha"],
        dropout=meta["dropout"],
        target_modules=tuple(meta["target_modules"]),
    )
    model, patched = apply_moe_lora(model, config)
    missing = [name for name in meta["patched_layers"] if name not in patched]
    if missing:
        raise RuntimeError(f"Failed to patch expected MoE-LoRA layers: {missing[:5]}")
    state = torch.load(adapter_path / "moe_lora_adapters.pt", map_location="cpu")
    for name, layer_state in state.items():
        module = model.get_submodule(name)
        if not isinstance(module, MoELoraLinear):
            continue
        _load_adapter_state_dict(module, layer_state)
    return model


def summarize_routing(model: nn.Module) -> dict[str, float]:
    """Mean routing entropy per MoE-LoRA layer (lower = sharper routing)."""
    summary: dict[str, float] = {}
    for name, module in model.named_modules():
        if isinstance(module, MoELoraLinear):
            # Router weight norm as a cheap sanity check when no batch is available.
            summary[name] = float(module.router.weight.norm().item())
    return summary
