"""Convert actor checkpoints whose ``mu`` head was a ``Sequential(Linear)``
into the current layout where ``mu`` is a bare ``Linear``.

Applies to ``actor.pt`` of both ``Actor`` and ``LiquidActor`` (same ``mu``
path). Parameters ``mu.0.weight`` / ``mu.0.bias`` become ``mu.weight`` /
``mu.bias``; the ``mu.0`` node (and any parameterless sibling such as a
trailing ``Tanh``) disappears from the module tree, which the C++
``InputArchive`` walks by hierarchy.

The parameter order is unchanged, so ``actor_optim.pt`` needs no conversion.

Usage:
    python convert_mu_linear.py outputs/train_501_ppo_liquid/save_*/actor.pt
    python convert_mu_linear.py --in-place save_9/actor.pt

Requires ``torch`` (CPU wheel is enough).
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn

MU_PARAM = re.compile(r"^mu\.0\.(weight|bias)$")


class Node(nn.Module):
    def forward(self) -> None:
        return None


def ensure_path(root: Node, dotted: str) -> nn.Module:
    module: nn.Module = root
    for part in dotted.split(".") if dotted else []:
        if part not in module._modules:  # pylint: disable=protected-access
            module.add_module(part, Node())
        module = module._modules[part]  # pylint: disable=protected-access
    return module


def convert_name(name: str) -> str:
    return MU_PARAM.sub(r"mu.\1", name)


def check_mu_layout(old_module: torch.jit.ScriptModule) -> None:
    mu_params = [
        n for n, _ in old_module.named_parameters() if n.startswith("mu.")
    ]
    if not mu_params:
        raise ValueError("no `mu.*` parameters found, not an actor checkpoint")
    if all(
        n.startswith("mu.weight") or n.startswith("mu.bias") for n in mu_params
    ):
        raise ValueError("`mu` is already a bare Linear, nothing to convert")
    unexpected = [n for n in mu_params if not MU_PARAM.match(n)]
    if unexpected:
        raise ValueError(f"unexpected parameters under `mu`: {unexpected}")


def convert(old_path: Path, new_path: Path) -> int:
    old_module = torch.jit.load(str(old_path), map_location="cpu")
    check_mu_layout(old_module)

    old_params: Dict[str, torch.Tensor] = {
        n: p.detach().clone() for n, p in old_module.named_parameters()
    }
    old_buffers: Dict[str, torch.Tensor] = {
        n: b.detach().clone() for n, b in old_module.named_buffers()
    }

    root = Node()
    dropped: List[str] = []
    for name, _ in old_module.named_modules():
        if not name:
            continue
        if name.startswith("mu."):
            dropped.append(name)
            continue
        ensure_path(root, name)
    for name, tensor in old_params.items():
        *path, leaf = convert_name(name).split(".")
        ensure_path(root, ".".join(path)).register_parameter(
            leaf, nn.Parameter(tensor)
        )
    for name, tensor in old_buffers.items():
        *path, leaf = convert_name(name).split(".")
        ensure_path(root, ".".join(path)).register_buffer(leaf, tensor)

    new_path.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(torch.jit.script(root), str(new_path))

    verify(new_path, old_params, old_buffers)
    print(f"{old_path} -> {new_path} (dropped modules: {', '.join(dropped)})")
    return len(old_params)


def verify(
    new_path: Path,
    old_params: Dict[str, torch.Tensor],
    old_buffers: Dict[str, torch.Tensor],
) -> None:
    reloaded = torch.jit.load(str(new_path), map_location="cpu")

    new_params = dict(reloaded.named_parameters())
    expected = {convert_name(n): t for n, t in old_params.items()}
    assert set(new_params) == set(expected), set(new_params) ^ set(expected)
    for name, tensor in expected.items():
        assert torch.equal(new_params[name], tensor), name

    new_buffers = dict(reloaded.named_buffers())
    expected_buffers = {convert_name(n): t for n, t in old_buffers.items()}
    assert set(new_buffers) == set(expected_buffers)
    for name, tensor in expected_buffers.items():
        assert torch.equal(new_buffers[name], tensor), name

    mu = reloaded.mu
    assert hasattr(mu, "weight") and hasattr(mu, "bias")
    assert not hasattr(mu, "0"), "`mu.0` still present"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "checkpoints",
        type=Path,
        nargs="+",
        help="actor.pt files saved with mu = Sequential(Linear)",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="overwrite each checkpoint, keeping the original as <name>.bak "
        "(default: write to a `converted/` sibling folder)",
    )
    args = parser.parse_args()

    for old_path in args.checkpoints:
        if args.in_place:
            backup = old_path.with_suffix(old_path.suffix + ".bak")
            old_path.rename(backup)
            convert(backup, old_path)
        else:
            convert(old_path, old_path.parent / "converted" / old_path.name)


if __name__ == "__main__":
    main()
