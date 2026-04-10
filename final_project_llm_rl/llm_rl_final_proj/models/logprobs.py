from __future__ import annotations

import torch
import torch.nn.functional as F


def compute_per_token_logprobs(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    enable_grad: bool = True,
) -> torch.Tensor:
    """Returns log p(x_t | x_<t) for t in [1, L-1]. Shape: [B, L-1]."""
    with torch.set_grad_enabled(enable_grad):
        # prev todo: run the causal LM, align logits with the next-token targets,
        # and return per-token log-probabilities of the observed tokens.
        # Hint: use F.cross_entropy with reduction='none' for memory efficiency.
        #raise NotImplementedError("Implement compute_per_token_logprobs in the student starter.")
        out = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = out.logits

        shift_logits = logits[:, :-1, :]  
        shift_targets = input_ids[:, 1:]  

        nll = F.cross_entropy(shift_logits.reshape(-1, shift_logits.size(-1)), shift_targets.reshape(-1), reduction="none")  
        return (-nll).reshape(shift_targets.shape)


def build_completion_mask(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_input_len: int,
    pad_token_id: int,
) -> torch.Tensor:
    """Mask over per-token positions [B, L-1], selecting completion tokens only."""
    del pad_token_id

    B, L = input_ids.shape

    shifted_attention_mask = attention_mask[:, 1:]

    positions = torch.arange(L - 1, device=input_ids.device)
    completed = (positions >= prompt_input_len - 1)

    return (shifted_attention_mask * completed.unsqueeze(0)).float()
    


def masked_sum(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum() / (mask.sum() + eps)


def masked_mean_per_row(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return (x * mask).sum(dim=1) / (mask.sum(dim=1) + eps)


def approx_kl_from_logprobs(
    new_logprobs: torch.Tensor,
    ref_logprobs: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1e-8,
    log_ratio_clip: float = 20.0,
) -> torch.Tensor:
    """Positive KL proxy from sampled actions.

    Uses estimator: exp(delta) - delta - 1 where delta = log p_ref(a) - log p_new(a).
    """
    del eps, log_ratio_clip
    # (student): implement the sampled-token KL proxy used throughout the codebase.
    # You should mask out non-completion positions and return a scalar batch mean.
    delta = ref_logprobs - new_logprobs


    kl_per_token = torch.exp(delta) - delta - 1

    masked_kl = kl_per_token * mask
    kl = masked_kl.sum() / (mask.sum() + 1e-8)
    return kl