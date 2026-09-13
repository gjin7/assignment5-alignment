"""Utilities for GRPO training."""

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
