from __future__ import annotations


def resolve_placement(cfg) -> str:
    return cfg.get("W", "every_block")


def iter_injection_points(W: str, n_layers: int, every_k: int = 2):
    if W == "early":
        yield ("early", None)
    elif W == "every_block":
        for i in range(n_layers):
            yield ("block", i)
    elif W == "late":
        yield ("late", None)
    elif W == "every_k":
        for i in range(n_layers):
            if (i + 1) % every_k == 0:
                yield ("block", i)
    else:
        raise ValueError(f"unknown placement W={W!r}")


__all__ = ["resolve_placement", "iter_injection_points"]
