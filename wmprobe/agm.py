import torch


def agm_loss(model, obs, act, eps=0.1):
    pred_clean = model(obs, act)
    pred_pert = model(obs, act + torch.randn_like(act) * eps)
    response = pred_pert - pred_clean
    return -(response.float() ** 2).mean()


def action_response(model, obs, act, eps=0.1):
    with torch.no_grad():
        return ((model(obs, act + torch.randn_like(act) * eps) - model(obs, act)) ** 2).mean().item()
