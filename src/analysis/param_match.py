from __future__ import annotations


def count_params(module, only_trainable=True):
    return sum(p.numel() for p in module.parameters() if (not only_trainable or p.requires_grad))


def within_match(total_a, total_b, tol=0.10):
    if total_a == 0 or total_b == 0:
        return False
    return abs(total_a - total_b) / max(total_a, total_b) <= tol


def scale_for_match(target_total, current_total):
    return target_total / max(current_total, 1)


__all__ = ["count_params", "within_match", "scale_for_match"]
