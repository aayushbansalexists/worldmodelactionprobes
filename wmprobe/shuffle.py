import torch


def shuffle_actions(act, generator=None):
    perm = torch.randperm(act.shape[0], device=act.device, generator=generator)
    return act[perm]


def difference_in_differences(m_method_logged, m_base_logged,
                              m_method_shuffled, m_base_shuffled):
    d_logged = m_method_logged - m_base_logged
    d_shuffled = m_method_shuffled - m_base_shuffled
    return d_logged, d_shuffled, d_logged - d_shuffled


def genuineness_ratio(gain_causal, gain_shuffled):
    if gain_causal <= 0:
        return 0.0
    return float(gain_shuffled / gain_causal)
