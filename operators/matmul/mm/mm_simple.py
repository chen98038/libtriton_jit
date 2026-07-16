import torch
import triton
from triton import language as tl


@triton.jit
def matmul_simple_kernel(
    A, B, C,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Simple matmul without tl.dot - use element-wise operations instead.
    This is slow but helps debug if tl.dot is the issue.
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Each program computes one element of C
    if pid_m < M and pid_n < N:
        # Compute C[pid_m, pid_n] = sum(A[pid_m, :] * B[:, pid_n])
        acc = tl.zeros((1,), dtype=tl.float32)

        for k in range(0, K, BLOCK_SIZE):
            k_idx = k + tl.arange(0, BLOCK_SIZE)
            mask = k_idx < K

            a_val = tl.load(A + pid_m * stride_am + k_idx * stride_ak, mask=mask, other=0.0)
            b_val = tl.load(B + k_idx * stride_bk + pid_n * stride_bn, mask=mask, other=0.0)

            acc += tl.sum(a_val * b_val)

        tl.store(C + pid_m * stride_cm + pid_n * stride_cn, acc)
