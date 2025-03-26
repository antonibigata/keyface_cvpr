import torch
from scipy import integrate
from einops import repeat, rearrange
from ...util import append_dims


def linear_multistep_coeff(order, t, i, j, epsrel=1e-4):
    if order - 1 > i:
        raise ValueError(f"Order {order} too high for step {i}")

    def fn(tau):
        prod = 1.0
        for k in range(order):
            if j == k:
                continue
            prod *= (tau - t[i - k]) / (t[i - j] - t[i - k])
        return prod

    return integrate.quad(fn, t[i], t[i + 1], epsrel=epsrel)[0]


def get_ancestral_step(sigma_from, sigma_to, eta=1.0):
    if not eta:
        return sigma_to, 0.0
    sigma_up = torch.minimum(
        sigma_to,
        eta * (sigma_to**2 * (sigma_from**2 - sigma_to**2) / sigma_from**2) ** 0.5,
    )
    sigma_down = (sigma_to**2 - sigma_up**2) ** 0.5
    return sigma_down, sigma_up


def to_d(x, sigma, denoised):
    return (x - denoised) / append_dims(sigma, x.ndim)


def to_neg_log_sigma(sigma):
    return sigma.log().neg()


def to_sigma(neg_log_sigma):
    return neg_log_sigma.neg().exp()


def chunk_inputs(
    input,
    cond,
    additional_model_inputs,
    sigma,
    sigma_next,
    start_idx,
    end_idx,
    num_frames=14,
):
    input_chunk = input[:, :, start_idx:end_idx].to(torch.float32).clone()

    sigma_chunk = sigma[start_idx:end_idx].to(torch.float32)
    sigma_next_chunk = sigma_next[start_idx:end_idx].to(torch.float32)

    cond_chunk = {}
    for k, v in cond.items():
        if isinstance(v, torch.Tensor):
            cond_chunk[k] = v[start_idx:end_idx]
        else:
            cond_chunk[k] = v

    additional_model_inputs_chunk = {}
    for k, v in additional_model_inputs.items():
        if isinstance(v, torch.Tensor):
            cond_chunk[k] = v[start_idx:end_idx]
        else:
            additional_model_inputs_chunk[k] = v

    return (
        input_chunk,
        sigma_chunk,
        sigma_next_chunk,
        cond_chunk,
        additional_model_inputs_chunk,
    )
