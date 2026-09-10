import pytest

from cs336_alignment import vllm_utils
from cs336_alignment.vllm_utils import VLLMCompletion
from scripts.evaluate_gsm8k import (
    PROMPT_CONFIGS,
    classify_reward,
    grade_completions,
    parse_ground_truth,
)


def test_parse_ground_truth() -> None:
    answer = "The reasoning is here.\n#### 1,234"
    assert parse_ground_truth(answer) == "1,234"


def test_parse_ground_truth_requires_delimiter() -> None:
    with pytest.raises(ValueError, match="####"):
        parse_ground_truth("The answer is 12.")


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        ({"format_reward": 1.0, "answer_reward": 1.0}, "correct"),
        (
            {"format_reward": 1.0, "answer_reward": 0.0},
            "formatted_incorrect",
        ),
        (
            {"format_reward": 0.0, "answer_reward": 0.0},
            "unformatted_incorrect",
        ),
    ],
)
def test_classify_reward(metrics: dict[str, float], expected: str) -> None:
    assert classify_reward(metrics) == expected


@pytest.mark.parametrize(
    ("prompt_type", "response"),
    [
        ("question_only", r"The final answer is \boxed{18}."),
        (
            "r1_zero",
            "Nine eggs remain, worth two dollars each. </think> <answer>18</answer>",
        ),
    ],
)
def test_grade_completions_uses_matching_grader(
    prompt_type: str, response: str
) -> None:
    results = grade_completions(
        examples=[{"question": "How much?", "answer": "Reasoning.\n#### 18"}],
        prompts=["prompt"],
        completions=[VLLMCompletion(response, [1, 2], "stop")],
        config=PROMPT_CONFIGS[prompt_type],
    )
    assert results[0]["category"] == "correct"


def test_generate_completions_passes_top_p(monkeypatch: pytest.MonkeyPatch) -> None:
    captured_payload = None

    def fake_http_json(method, url, payload, timeout):
        nonlocal captured_payload
        captured_payload = payload
        return {
            "choices": [
                {
                    "index": 0,
                    "text": "response",
                    "token_ids": [1],
                    "finish_reason": "stop",
                }
            ]
        }

    monkeypatch.setattr(vllm_utils, "_http_json", fake_http_json)
    vllm_utils.generate_completions(
        vllm_base_url="http://localhost:8000",
        model_id="model",
        prompts=["prompt"],
        sampling_params={
            "temperature": 1.0,
            "top_p": 0.9,
            "max_tokens": 512,
            "n": 1,
            "seed": 0,
        },
    )
    assert captured_payload is not None
    assert captured_payload["top_p"] == 0.9
