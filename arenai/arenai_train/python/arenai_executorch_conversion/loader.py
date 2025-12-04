import json
import os

import numpy as np
import torch
from torch import nn

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "uint8": torch.uint8,
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "int64": torch.int64,
    "bool": torch.bool,
}

NP_MAP = {
    "float32": np.float32,
    "float64": np.float64,
    "float16": np.float16,
    "bfloat16": np.float16,
    "uint8": np.uint8,
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "int64": np.int64,
    "bool": np.bool_,
}


def load_neutral_state_into(model: nn.Module, ckpt_dir: str):
    with open(os.path.join(ckpt_dir, "manifest.json"), "r") as f:
        manifest = json.load(f)
    sd = model.state_dict()
    missing = []
    for t in manifest["tensors"]:
        name = t["name"]
        path = os.path.join(ckpt_dir, t["file"])
        dtype = t["dtype"]
        shape = t["shape"]
        if name not in sd:
            missing.append(name)
            continue
        arr: np.ndarray = np.fromfile(path, dtype=NP_MAP[dtype])
        if np.prod(shape) != arr.size:
            raise RuntimeError(
                f"Shape mismatch for {name}: file has {arr.size} elements, expected {np.prod(shape)}."
            )
        arr = arr.reshape(shape)
        ten = torch.from_numpy(arr).to(DTYPE_MAP[dtype])

        try:
            sd[name].copy_(ten)
        except RuntimeError as e:
            raise RuntimeError(f"{name} failed to load") from e
    model.load_state_dict(sd, strict=False)
    if missing:
        print("[warn] missing tensors in model:", missing)
