from typing import *
import torch
from .. import SparseTensor
from .. import DEBUG, ATTN

if ATTN == 'xformers':
    import xformers.ops as xops
elif ATTN == 'flash_attn':
    import flash_attn
elif ATTN == 'sdpa':
    from torch.nn.attention.flex_attention import create_block_mask
    from .. import flex_attention
else:
    raise ValueError(f"Unknown attention module: {ATTN}")


__all__ = [
    'sparse_scaled_dot_product_attention',
]


@overload
def sparse_scaled_dot_product_attention(qkv: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        qkv (SparseTensor): A [N, *, 3, H, C] sparse tensor containing Qs, Ks, and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, kv: Union[SparseTensor, torch.Tensor]) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, C] sparse tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor or a [N, L, 2, H, C] dense tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, kv: SparseTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, L, H, C] dense tensor containing Qs.
        kv (SparseTensor or torch.Tensor): A [N, *, 2, H, C] sparse tensor containing Ks and Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, k: SparseTensor, v: SparseTensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.

    Note:
        k and v are assumed to have the same coordinate map.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: SparseTensor, k: torch.Tensor, v: torch.Tensor) -> SparseTensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (SparseTensor): A [N, *, H, Ci] sparse tensor containing Qs.
        k (torch.Tensor): A [N, L, H, Ci] dense tensor containing Ks.
        v (torch.Tensor): A [N, L, H, Co] dense tensor containing Vs.
    """
    ...

@overload
def sparse_scaled_dot_product_attention(q: torch.Tensor, k: SparseTensor, v: SparseTensor) -> torch.Tensor:
    """
    Apply scaled dot product attention to a sparse tensor.

    Args:
        q (torch.Tensor): A [N, L, H, Ci] dense tensor containing Qs.
        k (SparseTensor): A [N, *, H, Ci] sparse tensor containing Ks.
        v (SparseTensor): A [N, *, H, Co] sparse tensor containing Vs.
    """
    ...

def sparse_scaled_dot_product_attention(*args, **kwargs):
    arg_names_dict = {
        1: ['qkv'],
        2: ['q', 'kv'],
        3: ['q', 'k', 'v']
    }
    num_all_args = len(args) + len(kwargs)
    assert num_all_args in arg_names_dict, f"Invalid number of arguments, got {num_all_args}, expected 1, 2, or 3"
    for key in arg_names_dict[num_all_args][len(args):]:
        assert key in kwargs, f"Missing argument {key}"

    if num_all_args == 1:
        qkv = args[0] if len(args) > 0 else kwargs['qkv']
        assert isinstance(qkv, SparseTensor), f"qkv must be a SparseTensor, got {type(qkv)}"
        assert len(qkv.shape) == 4 and qkv.shape[1] == 3, f"Invalid shape for qkv, got {qkv.shape}, expected [N, *, 3, H, C]"
        device = qkv.device

        s = qkv
        q_seqlen = [qkv.layout[i].stop - qkv.layout[i].start for i in range(qkv.shape[0])]
        kv_seqlen = q_seqlen
        qkv = qkv.feats     # [T, 3, H, C]

    elif num_all_args == 2:
        q = args[0] if len(args) > 0 else kwargs['q']
        kv = args[1] if len(args) > 1 else kwargs['kv']
        assert isinstance(q, SparseTensor) and isinstance(kv, (SparseTensor, torch.Tensor)) or \
               isinstance(q, torch.Tensor) and isinstance(kv, SparseTensor), \
               f"Invalid types, got {type(q)} and {type(kv)}"
        assert q.shape[0] == kv.shape[0], f"Batch size mismatch, got {q.shape[0]} and {kv.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, C]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, C]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, C]"
            s = None
            N, L, H, C = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, C)   # [T_Q, H, C]

        if isinstance(kv, SparseTensor):
            assert len(kv.shape) == 4 and kv.shape[1] == 2, f"Invalid shape for kv, got {kv.shape}, expected [N, *, 2, H, C]"
            kv_seqlen = [kv.layout[i].stop - kv.layout[i].start for i in range(kv.shape[0])]
            kv = kv.feats     # [T_KV, 2, H, C]
        else:
            assert len(kv.shape) == 5, f"Invalid shape for kv, got {kv.shape}, expected [N, L, 2, H, C]"
            N, L, _, H, C = kv.shape
            kv_seqlen = [L] * N
            kv = kv.reshape(N * L, 2, H, C)   # [T_KV, 2, H, C]

    elif num_all_args == 3:
        q = args[0] if len(args) > 0 else kwargs['q']
        k = args[1] if len(args) > 1 else kwargs['k']
        v = args[2] if len(args) > 2 else kwargs['v']
        assert isinstance(q, SparseTensor) and isinstance(k, (SparseTensor, torch.Tensor)) and type(k) == type(v) or \
               isinstance(q, torch.Tensor) and isinstance(k, SparseTensor) and isinstance(v, SparseTensor), \
               f"Invalid types, got {type(q)}, {type(k)}, and {type(v)}"
        assert q.shape[0] == k.shape[0] == v.shape[0], f"Batch size mismatch, got {q.shape[0]}, {k.shape[0]}, and {v.shape[0]}"
        device = q.device

        if isinstance(q, SparseTensor):
            assert len(q.shape) == 3, f"Invalid shape for q, got {q.shape}, expected [N, *, H, Ci]"
            s = q
            q_seqlen = [q.layout[i].stop - q.layout[i].start for i in range(q.shape[0])]
            q = q.feats     # [T_Q, H, Ci]
        else:
            assert len(q.shape) == 4, f"Invalid shape for q, got {q.shape}, expected [N, L, H, Ci]"
            s = None
            N, L, H, CI = q.shape
            q_seqlen = [L] * N
            q = q.reshape(N * L, H, CI)  # [T_Q, H, Ci]

        if isinstance(k, SparseTensor):
            assert len(k.shape) == 3, f"Invalid shape for k, got {k.shape}, expected [N, *, H, Ci]"
            assert len(v.shape) == 3, f"Invalid shape for v, got {v.shape}, expected [N, *, H, Co]"
            kv_seqlen = [k.layout[i].stop - k.layout[i].start for i in range(k.shape[0])]
            k = k.feats     # [T_KV, H, Ci]
            v = v.feats     # [T_KV, H, Co]
        else:
            assert len(k.shape) == 4, f"Invalid shape for k, got {k.shape}, expected [N, L, H, Ci]"
            assert len(v.shape) == 4, f"Invalid shape for v, got {v.shape}, expected [N, L, H, Co]"
            N, L, H, CI, CO = *k.shape, v.shape[-1]
            kv_seqlen = [L] * N
            k = k.reshape(N * L, H, CI)     # [T_KV, H, Ci]
            v = v.reshape(N * L, H, CO)     # [T_KV, H, Co]

    if DEBUG:
        if s is not None:
            for i in range(s.shape[0]):
                assert (s.coords[s.layout[i]] == i).all(), f"SparseScaledDotProductSelfAttention: batch index mismatch"
        if num_all_args in [2, 3]:
            assert q.shape[:2] == [1, sum(q_seqlen)], f"SparseScaledDotProductSelfAttention: q shape mismatch"
        if num_all_args == 3:
            assert k.shape[:2] == [1, sum(kv_seqlen)], f"SparseScaledDotProductSelfAttention: k shape mismatch"
            assert v.shape[:2] == [1, sum(kv_seqlen)], f"SparseScaledDotProductSelfAttention: v shape mismatch"

    if ATTN == 'xformers':
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)
        mask = xops.fmha.BlockDiagonalMask.from_seqlens(q_seqlen, kv_seqlen)
        out = xops.memory_efficient_attention(q, k, v, mask)[0]
    elif ATTN == 'flash_attn':
        cu_seqlens_q = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(q_seqlen), dim=0)]).int().to(device)
        if num_all_args in [2, 3]:
            cu_seqlens_kv = torch.cat([torch.tensor([0]), torch.cumsum(torch.tensor(kv_seqlen), dim=0)]).int().to(device)
        if num_all_args == 1:
            out = flash_attn.flash_attn_varlen_qkvpacked_func(qkv, cu_seqlens_q, max(q_seqlen))
        elif num_all_args == 2:
            out = flash_attn.flash_attn_varlen_kvpacked_func(q, kv, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
        elif num_all_args == 3:
            out = flash_attn.flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_kv, max(q_seqlen), max(kv_seqlen))
    elif ATTN == 'sdpa':
        # 1. 解包参数 (同 xformers/sdpa)
        if num_all_args == 1:
            q, k, v = qkv.unbind(dim=1)
        elif num_all_args == 2:
            k, v = kv.unbind(dim=1)
        
        # 2. 调整维度: Input (L, H, D) -> FlexAttn (B=1, H, L, D)
        # flex_attention 也是以此为标准输入的，我们将 packed 序列视为 batch=1
        q_in = q.permute(1, 0, 2).unsqueeze(0)
        k_in = k.permute(1, 0, 2).unsqueeze(0)
        v_in = v.permute(1, 0, 2).unsqueeze(0)
        B, H, Q_L, D = q_in.shape
        KV_L = k_in.shape[2]
        # 3. 构建文档索引映射 (Document ID Map)
        # Q 和 K 都是拼接的，我们需要一个 Tensor 来标记每个 token 属于第几个样本
        # 例如 q_seqlen=[2, 3] -> doc_ids_q=[0, 0, 1, 1, 1]
        # 这一步是为了给 mask_mod 提供判断依据
        
        # 辅助函数：根据长度列表生成 ID 索引
        def create_doc_idx(lengths, total_len):
            # lengths 需要是 tensor 才能用于 repeat_interleave (或者转为 tensor)
            len_tensor = torch.as_tensor(lengths, device=device)
            ids = torch.arange(len(lengths), device=device, dtype=torch.int32)
            return torch.repeat_interleave(ids, len_tensor)
        doc_ids_q = create_doc_idx(q_seqlen, Q_L)
        doc_ids_k = create_doc_idx(kv_seqlen, KV_L)
        # 4. 定义 Mask 逻辑 (Block Diagonal Mask)
        # 核心逻辑：只有当 query 和 key 属于同一个文档(样本)时，mask 才为 true
        def document_mask_mod(b, h, q_idx, kv_idx):
            return doc_ids_q[q_idx] == doc_ids_k[kv_idx]
        # 5. 创建 Block Mask
        # create_block_mask 会分析 document_mask_mod，生成高效的稀疏掩码元数据
        block_mask = create_block_mask(
            document_mask_mod, 
            B=B, H=H, Q_LEN=Q_L, KV_LEN=KV_L, 
            device=device
        )
        # 6. 执行 Flex Attention
        out = flex_attention(q_in, k_in, v_in, block_mask=block_mask)
        # 7. 恢复维度: (1, H, L, D) -> (L, H, D)
        out = out.squeeze(0).permute(1, 0, 2)
    else:
        raise ValueError(f"Unknown attention module: {ATTN}")
    
    if s is not None:
        return s.replace(out)
    else:
        return out.reshape(N, L, H, -1)
