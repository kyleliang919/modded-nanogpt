import triton
import triton.language as tl
import torch
from torch.optim.optimizer import Optimizer

@triton.jit
def fused_c_adamw_kernel(
    param_ptr, grad_ptr, m_ptr, v_ptr,
    lr, beta1, beta2, eps, weight_decay,
    step, correct_bias,
    BLOCK_SIZE: tl.constexpr,
    n_elements: int
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    p = tl.load(param_ptr + offsets, mask=mask)
    g = tl.load(grad_ptr + offsets, mask=mask)
    m = tl.load(m_ptr + offsets, mask=mask)
    v = tl.load(v_ptr + offsets, mask=mask)

    # Update moments
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * g * g

    if correct_bias:
        step_f = tl.full([1], step, tl.float32)
        bias_correction1 = 1.0 - beta1 ** step_f
        bias_correction2 = 1.0 - beta2 ** step_f
        m_hat = m_new / bias_correction1
        v_hat = v_new / bias_correction2
    else:
        m_hat = m_new
        v_hat = v_new

    denom = tl.sqrt(v_hat) + eps

    # Gradient normalization trick from HF
    mask_pos = (m_hat * g > 0).to(tl.float32)
    mask_norm = mask_pos / tl.maximum(mask_pos.sum(), 1.0)
    update = (m_hat * mask_norm) / denom

    # Apply weight decay
    update += weight_decay * p

    # Update param
    p_new = p - lr * update

    # Store
    tl.store(param_ptr + offsets, p_new, mask=mask)
    tl.store(m_ptr + offsets, m_new, mask=mask)
    tl.store(v_ptr + offsets, v_new, mask=mask)

class FusedCAdamW(Optimizer):
    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        correct_bias=True
    ):
        defaults = dict(
            lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias
        )
        super().__init__(params, defaults)
        self._step = 0

    @torch.no_grad()
    def step(self, closure=None):
        loss = closure() if closure is not None else None
        self._step += 1

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError("FusedHFAdamW does not support sparse gradients.")

                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)

                m, v = state["exp_avg"], state["exp_avg_sq"]

                n = p.numel()
                BLOCK_SIZE = 1024
                grid = lambda META: (triton.cdiv(n, META['BLOCK_SIZE']),)

                fused_c_adamw_kernel[grid](
                    p, grad, m, v,
                    group["lr"],
                    group["betas"][0],
                    group["betas"][1],
                    group["eps"],
                    group["weight_decay"],
                    self._step,
                    int(group["correct_bias"]),
                    BLOCK_SIZE=BLOCK_SIZE,
                    n_elements=n
                )
        return loss