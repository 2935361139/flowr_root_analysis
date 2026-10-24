"""Reused, previously exercised ROOT replay utilities; no training or hooks."""
import random
from collections.abc import Mapping

def move_copy(value, device, torch):
    if torch.is_tensor(value):
        return value.to(device).clone()
    if isinstance(value, Mapping):
        return {k: move_copy(v, device, torch) for k, v in value.items()}
    if isinstance(value, list):
        return [move_copy(v, device, torch) for v in value]
    if isinstance(value, tuple):
        return tuple(move_copy(v, device, torch) for v in value)
    return value

def restore_rng(saved, device, torch, np):
    random.setstate(saved["python"])
    n = saved["numpy"]
    np.random.set_state((n[0], n[1].cpu().numpy().astype(np.uint32), int(n[2]), int(n[3]), float(n[4])))
    torch.set_rng_state(saved["torch_cpu"].cpu())
    if "torch_cuda" in saved:
        if device.type != "cuda":
            raise RuntimeError("Original forward ran on CUDA; use the same working GPU environment for replay.")
        torch.cuda.set_rng_state(saved["torch_cuda"].cpu(), device=device)

def tensor_leaves(value, torch, prefix=""):
    if torch.is_tensor(value):
        return {prefix: value}
    result = {}
    if isinstance(value, Mapping):
        for k, v in value.items():
            result.update(tensor_leaves(v, torch, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(value, (list, tuple)):
        for k, v in enumerate(value):
            result.update(tensor_leaves(v, torch, f"{prefix}.{k}"))
    return result

def compare_predictions(actual, expected, torch, np, atol, rtol):
    a, b = tensor_leaves(actual, torch), tensor_leaves(expected, torch)
    fields = {}
    required = {"coords", "atomics", "bonds", "charges", "mask"}
    passed = required.issubset(a) and required.issubset(b) and set(a) == set(b)
    for name in sorted(set(a) | set(b)):
        if name not in a or name not in b or tuple(a[name].shape) != tuple(b[name].shape):
            fields[name] = {"passed": False, "reason": "missing tensor or shape mismatch"}
            passed = False
            continue
        x = a[name].detach().cpu().float().numpy().astype(np.float64)
        y = b[name].detach().cpu().float().numpy().astype(np.float64)
        finite = bool(np.isfinite(x).all() and np.isfinite(y).all())
        close = finite and bool(np.allclose(x, y, atol=atol, rtol=rtol))
        changes = None
        if name in {"atomics", "bonds", "charges", "hybridization"}:
            changes = int(np.count_nonzero(x.argmax(-1) != y.argmax(-1)))
            close = close and changes == 0
        maximum = float(np.max(np.abs(x-y))) if x.size and finite else (0.0 if not x.size else None)
        fields[name] = {"passed": bool(close), "max_abs_difference": maximum,
                        "argmax_changes": changes, "finite": finite}
        passed = passed and close
    return {"passed": bool(passed), "atol": atol, "rtol": rtol,
            "note": "Coordinates are in model units; all tensor outputs are compared. Molecular categorical argmax decisions must also be identical.",
            "fields": fields}

def forward_once(model, call, pocket, encoding, device, torch, np):
    # Fresh copies matter: earlier computations may mutate a conditional tensor.
    batch = move_copy(call["state_before_forward"], device, torch)
    condition = move_copy(call["self_condition"], device, torch)
    times = move_copy(call["times"], device, torch)
    restore_rng(call["rng_before_forward"], device, torch, np)
    with torch.no_grad():
        return model(batch, pocket, times, cond_batch=condition,
                     pocket_equis=encoding["pocket_equis"], pocket_invs=encoding["pocket_invs"],
                     training=False)
