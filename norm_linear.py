#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Function
from contextlib import nullcontext

def clip_tensor_by_norm(tensor, max_norm, norm_type=2.0):
    """Clips a tensor's values by its norm."""
    tensor_norm = torch.linalg.norm(tensor, ord=norm_type, keepdim=True, dim = [1,2])
    clip_coef = max_norm / (tensor_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    return tensor * clip_coef_clamped

# -------------------- Custom Linear (vanilla behavior, ND-safe, autocast-safe) --------------------
class _LinearFn(Function):
    @staticmethod
    def forward(ctx, x, weight, bias=None):
        """
        x:      [..., in_features]
        weight: [out_features, in_features]
        bias:   [out_features] or None
        """
        ctx.save_for_backward(x, weight, bias if bias is not None else x.new_empty(0))
        ctx.has_bias = bias is not None
        # Use F.linear for numerics/autocast parity with nn.Linear
        return F.linear(x, weight, bias)

    @staticmethod
    def backward(ctx, grad_out):
        x, weight, _ = ctx.saved_tensors
        has_bias = ctx.has_bias
        out_features, in_features = weight.shape

        # Flatten leading dims for parameter grads
        grad_out_flat = grad_out.reshape(-1, out_features)
        x_flat = x.reshape(-1, in_features)

        # Compute in parameter dtype (usually fp32), then cast back for grad_x
        p_dtype = weight.dtype
        x_dtype = x.dtype

        # dL/dx must match input dtype
        grad_x = grad_out_flat.to(p_dtype).matmul(weight)      # [N*, in]
        grad_x = grad_x.to(x_dtype).view_as(x)

        # dL/dW and dL/db must match parameter dtype
        # grad_weight = grad_out_flat.to(p_dtype).t().matmul(x_flat.to(p_dtype))  # [out, in]
        # grad_weight = (grad_out_flat.to(p_dtype).unsqueeze(-1) @ x_flat.to(p_dtype).unsqueeze(-2)).sum(0)
        grad_weight = (grad_out_flat.to(p_dtype).unsqueeze(-1) @ x_flat.to(p_dtype).unsqueeze(-2))
        grad_weight = clip_tensor_by_norm(grad_weight, 1.0).mean(0)
        grad_bias = grad_out_flat.to(p_dtype).sum(dim=0) if has_bias else None

        return grad_x, grad_weight, grad_bias


class LinearVanilla(nn.Module):
    """Drop-in Linear implemented via custom autograd; matches nn.Linear behavior."""
    __constants__ = ["in_features", "out_features", "bias"]

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        # Match nn.Linear init
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / fan_in ** 0.5 if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return _LinearFn.apply(x, self.weight, self.bias)


# -------------------- Simple Parity Test Harness --------------------
def run_once(shape_prefix, in_f, out_f, use_bias, device="cpu", autocast_dtype=None, seed=0, tol_fwd=5e-6, tol_grad=5e-6):
    """
    shape_prefix: tuple for leading dims, so x has shape [..., in_f]
    autocast_dtype: None (no autocast) or torch.float16 / torch.bfloat16 on CUDA
    """
    g = torch.Generator(device=device).manual_seed(seed)
    x = torch.randn(*shape_prefix, in_f, device=device, dtype=torch.float32, generator=g, requires_grad=True)

    ref = nn.Linear(in_f, out_f, bias=use_bias).to(device=device, dtype=torch.float32)
    cus = LinearVanilla(in_f, out_f, bias=use_bias).to(device=device, dtype=torch.float32)

    # Copy parameters so they're identical
    cus.weight.data.copy_(ref.weight.data)
    if use_bias:
        cus.bias.data.copy_(ref.bias.data)

    # Same random target for both
    # (Use float32; autocast will handle compute dtypes)
    target = torch.randn(*shape_prefix, out_f, device=device, dtype=torch.float32, generator=g)

    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    ctx = (
        torch.autocast(device_type="cuda", dtype=autocast_dtype)  # half/bf16 math on CUDA
        if (device == "cuda" and autocast_dtype is not None)
        else nullcontext()
    )

    # ---------- Forward ----------
    with ctx:
        y_ref = ref(x)
        y_cus = cus(x)
    fwd_diff = (y_ref - y_cus).abs().max().item()

    # ---------- Param grads ----------
    ref.zero_grad(set_to_none=True)
    cus.zero_grad(set_to_none=True)
    with ctx:
        loss_ref = ((y_ref - target) ** 2).mean()
        loss_cus = ((y_cus - target) ** 2).mean()
    loss_ref.backward(retain_graph=True)
    loss_cus.backward(retain_graph=True)

    dW = (ref.weight.grad - cus.weight.grad).abs().max().item()
    dB = (ref.bias.grad - cus.bias.grad).abs().max().item() if use_bias else 0.0

    # ---------- Input grads ----------
    # Use separate clones so each graph is independent
    x_r = x.detach().clone().requires_grad_(True)
    x_c = x.detach().clone().requires_grad_(True)
    with ctx:
        yr = ref(x_r)
        yc = cus(x_c)
        lr = ((yr - target) ** 2).mean()
        lc = ((yc - target) ** 2).mean()
    gx_ref = torch.autograd.grad(lr, x_r, retain_graph=False)[0]
    gx_cus = torch.autograd.grad(lc, x_c, retain_graph=False)[0]
    dX = (gx_ref - gx_cus).abs().max().item()

    ok = (fwd_diff <= tol_fwd) and (dW <= tol_grad) and (dB <= tol_grad) and (dX <= tol_grad)
    return {
        "ok": ok,
        "fwd_max_abs": fwd_diff,
        "dW_max_abs": dW,
        "dB_max_abs": dB,
        "dX_max_abs": dX,
        "autocast": str(autocast_dtype),
        "shape": tuple(list(shape_prefix) + [in_f]),
        "bias": use_bias,
        "device": device,
    }


def main():
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    shapes = [
        (8,),         # [N, D]
        (2, 5),       # [B, T, D]
        (1, 3, 4),    # [A, B, C, D]
    ]
    in_f, out_f = 16, 13
    dtypes_to_try = [None]  # always test without autocast
    if device == "cuda":
        # also test with autocast half type
        if torch.cuda.is_bf16_supported():
            dtypes_to_try.append(torch.bfloat16)
        else:
            dtypes_to_try.append(torch.float16)

    results = []
    for shape in shapes:
        for bias in (False, True):
            for ac_dtype in dtypes_to_try:
                r = run_once(
                    shape_prefix=shape,
                    in_f=in_f,
                    out_f=out_f,
                    use_bias=bias,
                    device=device,
                    autocast_dtype=ac_dtype,
                    tol_fwd=5e-6 if ac_dtype is None else 5e-4,
                    tol_grad=5e-6 if ac_dtype is None else 5e-4,
                )
                results.append(r)
                status = "OK " if r["ok"] else "FAIL"
                print(
                    f"[{status}] shape={r['shape']} bias={r['bias']} device={r['device']} autocast={r['autocast']}"
                    f" | fwd={r['fwd_max_abs']:.3e} dW={r['dW_max_abs']:.3e} dB={r['dB_max_abs']:.3e} dX={r['dX_max_abs']:.3e}"
                )

    # Non-zero exit if any failure
    if not all(r["ok"] for r in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
