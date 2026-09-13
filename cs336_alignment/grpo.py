"""Utilities for GRPO training."""

from collections.abc import Callable
from typing import Literal

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase


def tokenize_prompt_and_output(
    prompt_strs: list[str],
    output_strs: list[str],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, torch.Tensor]:
    """Tokenize prompt/response pairs and align a response mask with labels."""
    if len(prompt_strs) != len(output_strs):
        raise ValueError("prompt_strs and output_strs must have the same length.")
    if not prompt_strs:
        raise ValueError("prompt_strs and output_strs must not be empty.")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must define a pad token.")

    prompt_token_ids = tokenizer(
        prompt_strs,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]
    output_token_ids = tokenizer(
        output_strs,
        add_special_tokens=False,
        return_attention_mask=False,
    )["input_ids"]

    combined_token_ids = [
        prompt_ids + response_ids
        for prompt_ids, response_ids in zip(
            prompt_token_ids, output_token_ids, strict=True
        )
    ]
    max_length = max(len(token_ids) for token_ids in combined_token_ids)

    padded_token_ids = torch.full(
        (len(combined_token_ids), max_length),
        fill_value=tokenizer.pad_token_id,
        dtype=torch.long,
    )
    token_is_response = torch.zeros_like(padded_token_ids, dtype=torch.bool)

    for row, (prompt_ids, response_ids) in enumerate(
        zip(prompt_token_ids, output_token_ids, strict=True)
    ):
        combined_ids = prompt_ids + response_ids
        combined_length = len(combined_ids)
        prompt_length = len(prompt_ids)
        padded_token_ids[row, :combined_length] = torch.tensor(combined_ids)
        token_is_response[row, prompt_length:combined_length] = True

    return {
        "input_ids": padded_token_ids[:, :-1],
        "labels": padded_token_ids[:, 1:],
        "response_mask": token_is_response[:, 1:],
    }


def get_response_log_probs(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    return_token_entropy: bool = False,
) -> dict[str, torch.Tensor]:
    """Score each label token under the model's next-token distribution."""
    if input_ids.shape != labels.shape:
        raise ValueError("input_ids and labels must have the same shape.")

    logits = model(input_ids=input_ids).logits
    all_log_probs = torch.log_softmax(logits, dim=-1, dtype=torch.float32)
    label_log_probs = torch.gather(
        all_log_probs,
        dim=-1,
        index=labels.unsqueeze(-1),
    ).squeeze(-1)

    result = {"log_probs": label_log_probs}
    if return_token_entropy:
        probabilities = all_log_probs.exp()
        result["token_entropy"] = -(probabilities * all_log_probs).sum(dim=-1)
    return result


def compute_rollout_rewards(
    reward_fn: Callable[[str, str], dict[str, float]],
    rollout_responses: list[str],
    repeated_ground_truths: list[str],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Score rollout responses and summarize their reward components."""
    if len(rollout_responses) != len(repeated_ground_truths):
        raise ValueError(
            "rollout_responses and repeated_ground_truths must have the same length."
        )
    if not rollout_responses:
        raise ValueError("rollout_responses must not be empty.")

    reward_results = [
        reward_fn(response, ground_truth)
        for response, ground_truth in zip(
            rollout_responses,
            repeated_ground_truths,
            strict=True,
        )
    ]

    raw_rewards = torch.tensor(
        [result["reward"] for result in reward_results],
        dtype=torch.float32,
    )
    metadata = {
        "reward_mean": raw_rewards.mean().item(),
        "format_reward_mean": sum(
            result["format_reward"] for result in reward_results
        )
        / len(reward_results),
        "answer_reward_mean": sum(
            result["answer_reward"] for result in reward_results
        )
        / len(reward_results),
    }
    return raw_rewards, metadata


def compute_group_normalized_rewards(
    raw_rewards: torch.Tensor,
    group_size: int,
    baseline: Literal["mean", "none"] = "mean",
    advantage_eps: float = 1e-6,
    advantage_normalizer: Literal["std", "mean", "none"] = "std",
) -> tuple[torch.Tensor, dict[str, float]]:
    """Convert each group's raw rewards into normalized advantages."""
    if baseline != "mean":
        raise NotImplementedError(f"Unsupported baseline: {baseline}")
    if advantage_normalizer != "std":
        raise NotImplementedError(
            f"Unsupported advantage normalizer: {advantage_normalizer}"
        )
    if raw_rewards.ndim != 1:
        raise ValueError("raw_rewards must be a one-dimensional tensor.")
    if group_size <= 0:
        raise ValueError("group_size must be positive.")
    if raw_rewards.numel() == 0:
        raise ValueError("raw_rewards must not be empty.")
    if raw_rewards.numel() % group_size != 0:
        raise ValueError("The number of rewards must be divisible by group_size.")

    grouped_rewards = raw_rewards.reshape(-1, group_size)
    group_means = grouped_rewards.mean(dim=1, keepdim=True)
    if group_size == 1:
        group_stds = torch.zeros_like(group_means)
    else:
        group_stds = grouped_rewards.std(dim=1, keepdim=True, correction=1)

    grouped_advantages = (grouped_rewards - group_means) / (
        group_stds + advantage_eps
    )
    advantages = grouped_advantages.flatten()
    metadata = {
        "advantage_mean": advantages.mean().item(),
        "advantage_std": advantages.std(correction=0).item(),
    }
    return advantages, metadata


def compute_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    importance_reweighting_method: Literal[
        "none", "noclip", "grpo", "gspo"
    ] = "none",
    old_log_probs: torch.Tensor | None = None,
    cliprange: float | None = None,
    response_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the on-policy policy-gradient loss at each token."""
    del old_log_probs, cliprange, response_mask

    if importance_reweighting_method != "none":
        raise NotImplementedError(
            "Importance reweighting is not implemented yet."
        )
    if policy_log_probs.ndim != 2:
        raise ValueError("policy_log_probs must have shape (batch_size, sequence_length).")
    if raw_rewards_or_advantages.ndim == 1:
        advantages = raw_rewards_or_advantages.unsqueeze(1)
    elif (
        raw_rewards_or_advantages.ndim == 2
        and raw_rewards_or_advantages.shape[1] == 1
    ):
        advantages = raw_rewards_or_advantages
    else:
        raise ValueError(
            "raw_rewards_or_advantages must have shape (batch_size,) "
            "or (batch_size, 1)."
        )
    if advantages.shape[0] != policy_log_probs.shape[0]:
        raise ValueError("Advantages and policy_log_probs must have the same batch size.")

    per_token_loss = -advantages * policy_log_probs
    return per_token_loss, {}
