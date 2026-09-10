"""Evaluate OLMo-2-0425-1B on GSM8K with the provided prompt styles."""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cs336_alignment.drgrpo_grader import (
    question_only_reward_fn,
    r1_zero_reward_fn,
)
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_ID = "allenai/OLMo-2-0425-1B"
DEFAULT_DATA_PATH = REPO_ROOT / "data" / "gsm8k" / "test.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "gsm8k_prompting_baselines"
PROMPT_DIR = REPO_ROOT / "cs336_alignment" / "prompts"

RewardFunction = Callable[[str, str], dict[str, float]]


@dataclass(frozen=True)
class PromptConfig:
    name: str
    template_path: Path
    reward_function: RewardFunction
    stop: list[str] | None


PROMPT_CONFIGS = {
    "question_only": PromptConfig(
        name="question_only",
        template_path=PROMPT_DIR / "question_only.prompt",
        reward_function=question_only_reward_fn,
        stop=None,
    ),
    "r1_zero": PromptConfig(
        name="r1_zero",
        template_path=PROMPT_DIR / "r1_zero.prompt",
        reward_function=r1_zero_reward_fn,
        stop=["</answer>"],
    ),
    "r1_zero_three_shot": PromptConfig(
        name="r1_zero_three_shot",
        template_path=PROMPT_DIR / "r1_zero_three_shot_gsm8k.prompt",
        reward_function=r1_zero_reward_fn,
        stop=["</answer>"],
    ),
}

CATEGORY_NAMES = {
    "correct": "format=1, answer=1",
    "formatted_incorrect": "format=1, answer=0",
    "unformatted_incorrect": "format=0, answer=0",
}


def parse_ground_truth(answer: str) -> str:
    """Extract the final answer after GSM8K's ``####`` delimiter."""
    rationale, separator, final_answer = answer.rpartition("####")
    if not separator or not rationale or not final_answer.strip():
        raise ValueError("Expected a GSM8K answer containing '#### {answer}'.")
    return final_answer.strip()


def load_examples(data_path: Path, limit: int | None) -> list[dict[str, str]]:
    examples = []
    with data_path.open() as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if limit is not None and len(examples) >= limit:
                break
            example = json.loads(line)
            if not isinstance(example.get("question"), str) or not isinstance(
                example.get("answer"), str
            ):
                raise TypeError(
                    f"{data_path}:{line_number} must contain string question and answer fields."
                )
            examples.append(example)
    if not examples:
        raise ValueError(f"No examples found in {data_path}.")
    return examples


def render_prompts(
    examples: Sequence[dict[str, str]], config: PromptConfig
) -> list[str]:
    template = config.template_path.read_text()
    return [template.format(question=example["question"]) for example in examples]


def classify_reward(metrics: dict[str, float]) -> str:
    format_reward = metrics["format_reward"]
    answer_reward = metrics["answer_reward"]
    if format_reward == 1.0 and answer_reward == 1.0:
        return "correct"
    if format_reward == 1.0 and answer_reward == 0.0:
        return "formatted_incorrect"
    if format_reward == 0.0 and answer_reward == 0.0:
        return "unformatted_incorrect"
    raise ValueError(
        "Unexpected reward combination: "
        f"format_reward={format_reward}, answer_reward={answer_reward}."
    )


def grade_completions(
    examples: Sequence[dict[str, str]],
    prompts: Sequence[str],
    completions: Sequence[VLLMCompletion],
    config: PromptConfig,
) -> list[dict[str, Any]]:
    if not (len(examples) == len(prompts) == len(completions)):
        raise ValueError("Examples, prompts, and completions must have equal lengths.")

    results = []
    for index, (example, prompt, completion) in enumerate(
        zip(examples, prompts, completions, strict=True)
    ):
        ground_truth = parse_ground_truth(example["answer"])
        metrics = config.reward_function(completion.text, ground_truth)
        results.append(
            {
                "index": index,
                "prompt_type": config.name,
                "question": example["question"],
                "ground_truth_response": example["answer"],
                "ground_truth": ground_truth,
                "prompt": prompt,
                "response": completion.text,
                "token_ids": completion.token_ids,
                "finish_reason": completion.finish_reason,
                "metrics": metrics,
                "category": classify_reward(metrics),
            }
        )
    return results


def write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
    with path.open("w") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def print_examples(
    prompt_name: str,
    results: Sequence[dict[str, Any]],
    examples_per_category: int,
) -> None:
    if examples_per_category == 0:
        return
    for category, description in CATEGORY_NAMES.items():
        matching_results = [
            result for result in results if result["category"] == category
        ]
        for result in matching_results[:examples_per_category]:
            print(f"\n[{prompt_name} | {description} | example {result['index']}]")
            print(f"Prompt:\n{result['prompt']}")
            print(f"Response:\n{result['response']}")
            print(f"Ground truth: {result['ground_truth']}")


def evaluate_prompt(
    server: VLLMServer,
    examples: Sequence[dict[str, str]],
    config: PromptConfig,
    batch_size: int,
    seed: int,
) -> list[dict[str, Any]]:
    prompts = render_prompts(examples, config)
    sampling_params: dict[str, Any] = {
        "temperature": 1.0,
        "top_p": 1.0,
        "max_tokens": 512,
        "n": 1,
        "seed": seed,
    }
    if config.stop is not None:
        sampling_params["stop"] = config.stop
        sampling_params["include_stop_str_in_output"] = True

    LOGGER.info("Generating %d responses for %s", len(prompts), config.name)
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params=sampling_params,
        batch_size=batch_size,
    )
    return grade_completions(examples, prompts, completions, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OLMo-2-0425-1B on GSM8K prompting baselines."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--prompt-types",
        nargs="+",
        choices=tuple(PROMPT_CONFIGS),
        default=list(PROMPT_CONFIGS),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--examples-per-category", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument(
        "--connect-to-existing-server",
        action="store_true",
        help="Use an existing vLLM server instead of launching one.",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive.")
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive.")
    if args.examples_per_category < 0:
        parser.error("--examples-per-category cannot be negative.")
    if not 0 < args.gpu_memory_utilization <= 1:
        parser.error("--gpu-memory-utilization must be in (0, 1].")
    return args


def main() -> None:
    args = parse_args()
    examples = load_examples(args.data_path, args.limit)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    server = VLLMServer(
        model_id=args.model_id,
        host=args.host,
        port=args.port,
        gpu=args.gpu,
        seed=args.seed,
        gpu_memory_utilization=args.gpu_memory_utilization,
        launch_server=not args.connect_to_existing_server,
    )

    summaries = {}
    try:
        server.start()
        for prompt_name in args.prompt_types:
            config = PROMPT_CONFIGS[prompt_name]
            results = evaluate_prompt(
                server=server,
                examples=examples,
                config=config,
                batch_size=args.batch_size,
                seed=args.seed,
            )
            counts = Counter(result["category"] for result in results)
            summary = {
                "prompt_type": prompt_name,
                "num_examples": len(results),
                "counts": {name: counts[name] for name in CATEGORY_NAMES},
                "accuracy": counts["correct"] / len(results),
            }
            summaries[prompt_name] = summary

            output_path = args.output_dir / f"{prompt_name}.jsonl"
            write_jsonl(output_path, results)
            print(json.dumps(summary, indent=2))
            print_examples(prompt_name, results, args.examples_per_category)
            LOGGER.info("Wrote per-example results to %s", output_path)
    finally:
        server.stop()

    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "data_path": str(args.data_path),
                "generation": {
                    "temperature": 1.0,
                    "top_p": 1.0,
                    "max_tokens": 512,
                    "n": 1,
                    "seed": args.seed,
                },
                "prompts": summaries,
            },
            indent=2,
        )
        + "\n"
    )
    LOGGER.info("Wrote aggregate metrics to %s", summary_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
