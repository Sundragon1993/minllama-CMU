from typing import Tuple
import torch


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """
    Helper function to reshape frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.

    Returns:
        torch.Tensor: Reshaped frequency tensor.

    Raises:
        AssertionError: If the frequency tensor doesn't match the expected shape.
        AssertionError: If the target tensor 'x' doesn't have the expected number of dimensions.
    """
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(shape)


def apply_rotary_emb(
        query: torch.Tensor,
        key: torch.Tensor,
        head_dim: int,
        max_seq_len: int,
        theta: float = 10000.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor.

    This function applies rotary embeddings to the given query and key tensors. The rotation to each token
    embedding is a function of that token's position in the sequence, head_dim, and theta.
    The input tensors are reshaped as complex numbers to simplify your implementation.

    Args:
        query (torch.Tensor): Query tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_heads, self.head_dim)
        key (torch.Tensor): Key tensor to apply rotary embeddings.
                              Shape: (batch_size, seqlen, n_local_kv_heads, self.head_dim)
        head_dim (int): Dimension of each attention head.
        max_seq_len (int): Maximum sequence length supported by model.
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """

    _, seqlen, _, _ = query.shape
    device = query.device
    # Please refer to slide 22 in https://phontron.com/class/anlp2024/assets/slides/anlp-05-transformers.pdf
    # and Section 3 in https://arxiv.org/abs/2104.09864.

    # reshape xq and xk to match the complex representation
    query_real, query_imag = query.float().reshape(query.shape[:-1] + (-1, 2)).unbind(-1)
    key_real, key_imag = key.float().reshape(key.shape[:-1] + (-1, 2)).unbind(-1)
    # This separates each query/key vector into its odd and even indices (assuming *one-indexing*).
    # query_real contains q_1, q_3, q_5, ... and query_imag contains q_2, q_4, q_6, ...

    # Step 1: Precompute angles for rotary embeddings lucidrains
    freqs = 1. / (theta ** (torch.arange(0, head_dim, 2)[:(head_dim // 2)].float() / head_dim)).to(device)  # [2]

    # Step 2: Create position embedding indexes
    seq_idx = torch.arange(seqlen, device=device).float()[:max_seq_len].to(device) #[2] absolute position

    # idx_theta = torch.einsum('n,d->nd', seq_idx, theta)
    # idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
    # cos_cached = idx_theta2.cos()[:, None, None, :]
    # sin_cached = idx_theta2.sin()[:, None, None, :]

    # turn freqs to be put in reshape_for_broadcast -> freqs.shape == (query_real.shape[1], query_real.shape[-1])
    freqs = torch.outer(freqs, seq_idx).transpose(-2, -1).float()  # (2, 2)
    # query_real.shape (1, 2, 2, 2)
    freqs = reshape_for_broadcast(freqs, query_real)  # becomes -> (1, 2, 1, 2)
    # slide 22 from ppt


    query_rotated_real = query_real * freqs.cos() - query_imag * freqs.sin()
    query_rotated_imag = query_real * freqs.sin() + query_imag * freqs.cos()
    key_rotated_real = key_real * freqs.cos() - key_imag * freqs.sin()
    key_rotated_imag = key_real * freqs.sin() + key_imag * freqs.cos()

    # Then, combine these trigonometric values with the tensors query_real, query_imag,
    # key_real, and key_imag.
    # both (1, 2, 2, 2, 2)
    query_stack = torch.stack((query_rotated_real, query_rotated_imag), dim=-1)
    key_stack = torch.stack((key_rotated_real, key_rotated_imag), dim=-1)

    # turn to original shape -> both (1, 2, 2, 4)
    query_out = query_stack.reshape(query.shape)
    key_out = key_stack.reshape(key.shape)
    # Return the rotary position embeddings for the query and key tensors
    return query_out, key_out
