"""Train OLMo on GSM8K with standard on-policy GRPO."""

from __future__ import annotations

import argparse
import json
import logging
import random
import signal
import time
from pathlib import Path
from typing import Any

import torch

from cs336_alignment.checkpoint import get_model_and_tokenizer
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import grpo_train_step
from cs336_alignment.vllm_utils import VLLMCompletion, VLLMServer


LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_MODEL_ID = "allenai/OLMo-2-0425-1B"
DEFAULT_PROMPT_PATH = REPO_ROOT / "cs336_alignment" / "prompts" / "r1_zero.prompt"
DEFAULT_TRAIN_PATH = REPO_ROOT / "data" / "gsm8k" / "train.jsonl"
DEFAULT_VALIDATION_PATH = REPO_ROOT / "data" / "gsm8k" / "test.jsonl"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "grpo"


class GracefulShutdown(Exception):
    """Request an orderly shutdown after SIGINT or SIGTERM."""

    def __init__(self, signum: int):
        self.signum = signum
        super().__init__(signal.Signals(signum).name)


def handle_shutdown_signal(signum: int, _frame: Any) -> None:
    raise GracefulShutdown(signum)


def parse_ground_truth(answer: str) -> str:
    """Extract the final GSM8K answer after the ``####`` delimiter."""
    _, separator, final_answer = answer.rpartition("####")
    if not separator or not final_answer.strip():
        raise ValueError("Expected a GSM8K answer containing '#### {answer}'.")
    return final_answer.strip()


def load_gsm8k(path: Path) -> list[dict[str, str]]:
    """Load and validate GSM8K JSONL examples."""
    examples = []
    with path.open() as input_file:
        for line_number, line in enumerate(input_file, start=1):
            example = json.loads(line)
            if not isinstance(example.get("question"), str) or not isinstance(
                example.get("answer"), str
            ):
                raise TypeError(
                    f"{path}:{line_number} must contain string question and answer fields."
                )
            examples.append(example)
    if not examples:
        raise ValueError(f"No examples found in {path}.")
    return examples


def render_prompt(template: str, question: str) -> str:
    return template.format(question=question)


def generate_rollout_batch(
    server: VLLMServer,
    examples: list[dict[str, str]],
    prompt_template: str,
    group_size: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    request_batch_size: int,
) -> tuple[list[str], list[str], list[str], list[VLLMCompletion]]:
    """Generate ``group_size`` responses for every training prompt."""
    prompts = [
        render_prompt(prompt_template, example["question"])
        for example in examples
    ]
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": group_size,
            "seed": seed,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        },
        batch_size=request_batch_size,
    )

    expected_completions = len(examples) * group_size
    if len(completions) != expected_completions:
        raise RuntimeError(
            f"Expected {expected_completions} completions, got {len(completions)}."
        )

    repeated_prompts = []
    repeated_ground_truths = []
    for prompt, example in zip(prompts, examples, strict=True):
        ground_truth = parse_ground_truth(example["answer"])
        repeated_prompts.extend([prompt] * group_size)
        repeated_ground_truths.extend([ground_truth] * group_size)

    responses = [completion.text for completion in completions]
    return repeated_prompts, responses, repeated_ground_truths, completions


def evaluate(
    server: VLLMServer,
    tokenizer,
    examples: list[dict[str, str]],
    prompt_template: str,
    temperature: float,
    top_p: float,
    max_tokens: int,
    seed: int,
    request_batch_size: int,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Evaluate one sampled response for every validation example."""
    prompts = [
        render_prompt(prompt_template, example["question"])
        for example in examples
    ]
    completions = server.generate_completions(
        prompts=prompts,
        sampling_params={
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            "n": 1,
            "seed": seed,
            "stop": ["</answer>"],
            "include_stop_str_in_output": True,
        },
        batch_size=request_batch_size,
    )
    if len(completions) != len(examples):
        raise RuntimeError(
            f"Expected {len(examples)} validation completions, got {len(completions)}."
        )

    records = []
    reward_sum = 0.0
    format_reward_sum = 0.0
    response_token_count = 0

    for example, prompt, completion in zip(
        examples,
        prompts,
        completions,
        strict=True,
    ):
        ground_truth = parse_ground_truth(example["answer"])
        rewards = r1_zero_reward_fn(completion.text, ground_truth)
        token_count = len(completion.token_ids)
        if token_count == 0:
            token_count = len(
                tokenizer.encode(completion.text, add_special_tokens=False)
            )

        reward_sum += rewards["reward"]
        format_reward_sum += rewards["format_reward"]
        response_token_count += token_count
        records.append(
            {
                "question": example["question"],
                "prompt": prompt,
                "response": completion.text,
                "ground_truth": ground_truth,
                "reward": rewards["reward"],
                "format_reward": rewards["format_reward"],
                "answer_reward": rewards["answer_reward"],
                "response_length": token_count,
            }
        )

    num_examples = len(examples)
    metrics = {
        "val/reward": reward_sum / num_examples,
        "val/format_reward": format_reward_sum / num_examples,
        "val/average_response_length": response_token_count / num_examples,
    }
    return metrics, records


def make_training_rollout_records(
    examples: list[dict[str, str]],
    prompts: list[str],
    responses: list[str],
    ground_truths: list[str],
    completions: list[VLLMCompletion],
    group_size: int,
) -> list[dict[str, Any]]:
    """Build inspectable records for periodically logged training rollouts."""
    records = []
    for index, (prompt, response, ground_truth, completion) in enumerate(
        zip(prompts, responses, ground_truths, completions, strict=True)
    ):
        rewards = r1_zero_reward_fn(response, ground_truth)
        example = examples[index // group_size]
        records.append(
            {
                "question": example["question"],
                "rollout_index": index % group_size,
                "prompt": prompt,
                "response": response,
                "ground_truth": ground_truth,
                "reward": rewards["reward"],
                "format_reward": rewards["format_reward"],
                "answer_reward": rewards["answer_reward"],
                "response_length": len(completion.token_ids),
            }
        )
    return records


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w") as output_file:
        for record in records:
            output_file.write(json.dumps(record, ensure_ascii=False) + "\n")


def append_metrics(path: Path, step: int, metrics: dict[str, float]) -> None:
    record = {"step": step, **metrics}
    with path.open("a") as output_file:
        output_file.write(json.dumps(record) + "\n")


def write_run_status(
    run_dir: Path,
    status: str,
    last_completed_step: int,
    error: str | None = None,
) -> None:
    record: dict[str, str | int] = {
        "status": status,
        "last_completed_step": last_completed_step,
    }
    if error is not None:
        record["error"] = error
    (run_dir / "status.json").write_text(json.dumps(record, indent=2) + "\n")


def log_rollouts_to_wandb(
    wandb_run,
    split: str,
    step: int,
    records: list[dict[str, Any]],
    max_examples: int = 16,
) -> None:
    if wandb_run is None:
        return

    import wandb

    columns = [
        "question",
        "response",
        "ground_truth",
        "reward",
        "format_reward",
        "response_length",
    ]
    rows = [[record[column] for column in columns] for record in records[:max_examples]]
    wandb_run.log(
        {f"{split}/rollouts": wandb.Table(columns=columns, data=rows)},
        step=step,
    )


def initialize_wandb(args: argparse.Namespace, run_name: str):
    if args.wandb_mode == "disabled":
        return None

    import wandb

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    return wandb.init(
        project=args.wandb_project,
        name=run_name,
        group=args.wandb_group,
        mode=args.wandb_mode,
        config=config,
    )


def log_metrics(
    metrics_path: Path,
    wandb_run,
    step: int,
    metrics: dict[str, float],
) -> None:
    append_metrics(metrics_path, step, metrics)
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)
    LOGGER.info("step=%d metrics=%s", step, json.dumps(metrics, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train OLMo-2-0425-1B on GSM8K with on-policy GRPO."
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--prompt-path", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--train-data-path", type=Path, default=DEFAULT_TRAIN_PATH)
    parser.add_argument(
        "--validation-data-path",
        type=Path,
        default=DEFAULT_VALIDATION_PATH,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name")

    parser.add_argument("--n-train-examples", type=int, default=6400)
    parser.add_argument("--n-validation-examples", type=int, default=1024)
    parser.add_argument("--num-rollout-steps", type=int, default=200)
    parser.add_argument("--rollout-batch-size", type=int, default=256)
    parser.add_argument("--train-batch-size", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=32)

    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--sampling-temperature", type=float, default=1.0)
    parser.add_argument("--sampling-top-p", type=float, default=1.0)
    parser.add_argument("--sampling-max-tokens", type=int, default=512)

    parser.add_argument("--validation-interval", type=int, default=10)
    parser.add_argument("--rollout-log-interval", type=int, default=40)
    parser.add_argument("--vllm-request-batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--policy-gpu", type=int, default=0)
    parser.add_argument("--vllm-gpu", type=int, default=1)
    parser.add_argument("--vllm-port", type=int, default=8000)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=0.9)

    parser.add_argument("--wandb-project", default="cs336-a5-grpo")
    parser.add_argument("--wandb-group", default="standard-on-policy")
    parser.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default="offline",
    )
    parser.add_argument("--save-final-model", action="store_true")

    args = parser.parse_args()
    positive_integer_names = (
        "n_train_examples",
        "n_validation_examples",
        "num_rollout_steps",
        "rollout_batch_size",
        "train_batch_size",
        "group_size",
        "gradient_accumulation_steps",
        "sampling_max_tokens",
        "validation_interval",
        "rollout_log_interval",
        "vllm_request_batch_size",
    )
    for name in positive_integer_names:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive.")
    if args.rollout_batch_size != args.train_batch_size:
        parser.error(
            "Standard on-policy training requires --rollout-batch-size and "
            "--train-batch-size to match."
        )
    if args.rollout_batch_size % args.group_size != 0:
        parser.error("--rollout-batch-size must be divisible by --group-size.")
    if args.train_batch_size % args.gradient_accumulation_steps != 0:
        parser.error(
            "--train-batch-size must be divisible by "
            "--gradient-accumulation-steps."
        )
    if args.policy_gpu == args.vllm_gpu:
        parser.error("--policy-gpu and --vllm-gpu must select different GPUs.")
    if not 0.0 < args.sampling_top_p <= 1.0:
        parser.error("--sampling-top-p must be in (0, 1].")
    if args.sampling_temperature < 0.0:
        parser.error("--sampling-temperature must be non-negative.")
    if not 0.0 < args.vllm_gpu_memory_utilization <= 1.0:
        parser.error("--vllm-gpu-memory-utilization must be in (0, 1].")
    return args


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    train_examples = load_gsm8k(args.train_data_path)
    random.Random(args.seed).shuffle(train_examples)
    if args.n_train_examples > len(train_examples):
        raise ValueError(
            f"Requested {args.n_train_examples} training examples, but only "
            f"{len(train_examples)} are available."
        )
    train_examples = train_examples[: args.n_train_examples]

    validation_examples = load_gsm8k(args.validation_data_path)
    if args.n_validation_examples > len(validation_examples):
        raise ValueError(
            f"Requested {args.n_validation_examples} validation examples, but only "
            f"{len(validation_examples)} are available."
        )
    validation_examples = validation_examples[: args.n_validation_examples]

    prompts_per_step = args.rollout_batch_size // args.group_size
    required_train_examples = args.num_rollout_steps * prompts_per_step
    if required_train_examples > len(train_examples):
        raise ValueError(
            f"{args.num_rollout_steps} rollout steps require "
            f"{required_train_examples} training examples, but only "
            f"{len(train_examples)} were selected."
        )

    run_name = args.run_name
    if run_name is None:
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        run_name = f"standard_on_policy_seed{args.seed}_{timestamp}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.jsonl"

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n")

    prompt_template = args.prompt_path.read_text()
    policy_device = f"cuda:{args.policy_gpu}"
    server = VLLMServer(
        model_id=args.model_id,
        port=args.vllm_port,
        gpu=args.vllm_gpu,
        seed=args.seed,
        gpu_memory_utilization=args.vllm_gpu_memory_utilization,
    )
    wandb_run = initialize_wandb(args, run_name)
    last_completed_step = 0
    interrupted_signal = None

    try:
        signal.signal(signal.SIGINT, handle_shutdown_signal)
        signal.signal(signal.SIGTERM, handle_shutdown_signal)
        write_run_status(run_dir, status="running", last_completed_step=0)

        server.start()
        policy, tokenizer = get_model_and_tokenizer(args.model_id, policy_device)
        policy.config.use_cache = False
        optimizer = torch.optim.AdamW(
            policy.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.0,
        )
        server.init_weight_sync(policy_device)

        server.sync_policy_weights(policy)
        validation_metrics, validation_records = evaluate(
            server=server,
            tokenizer=tokenizer,
            examples=validation_examples,
            prompt_template=prompt_template,
            temperature=args.sampling_temperature,
            top_p=args.sampling_top_p,
            max_tokens=args.sampling_max_tokens,
            seed=args.seed,
            request_batch_size=args.vllm_request_batch_size,
        )
        log_metrics(metrics_path, wandb_run, step=0, metrics=validation_metrics)
        write_jsonl(run_dir / "validation_rollouts_step_0000.jsonl", validation_records)
        log_rollouts_to_wandb(
            wandb_run,
            split="val",
            step=0,
            records=validation_records,
        )

        for step in range(1, args.num_rollout_steps + 1):
            server.sync_policy_weights(policy)

            example_start = (step - 1) * prompts_per_step
            example_end = step * prompts_per_step
            step_examples = train_examples[example_start:example_end]
            repeated_prompts, responses, repeated_ground_truths, completions = (
                generate_rollout_batch(
                    server=server,
                    examples=step_examples,
                    prompt_template=prompt_template,
                    group_size=args.group_size,
                    temperature=args.sampling_temperature,
                    top_p=args.sampling_top_p,
                    max_tokens=args.sampling_max_tokens,
                    seed=args.seed + step,
                    request_batch_size=args.vllm_request_batch_size,
                )
            )

            loss, train_metadata = grpo_train_step(
                model=policy,
                tokenizer=tokenizer,
                optimizer=optimizer,
                gradient_accumulation_steps=args.gradient_accumulation_steps,
                max_grad_norm=args.max_grad_norm,
                reward_fn=r1_zero_reward_fn,
                repeated_prompts=repeated_prompts,
                rollout_responses=responses,
                repeated_ground_truths=repeated_ground_truths,
                group_size=args.group_size,
            )
            train_metrics = {
                "train/loss": loss.item(),
                "train/gradient_norm": float(train_metadata["gradient_norm"]),
                "train/token_entropy": float(train_metadata["token_entropy"]),
                "train/reward": float(train_metadata["reward_mean"]),
                "train/format_reward": float(
                    train_metadata["format_reward_mean"]
                ),
                "train/empty_response_fraction": sum(
                    response == "" for response in responses
                )
                / len(responses),
            }
            log_metrics(metrics_path, wandb_run, step=step, metrics=train_metrics)
            last_completed_step = step

            should_log_rollouts = (
                step == 1
                or step % args.rollout_log_interval == 0
                or step == args.num_rollout_steps
            )
            if should_log_rollouts:
                training_records = make_training_rollout_records(
                    examples=step_examples,
                    prompts=repeated_prompts,
                    responses=responses,
                    ground_truths=repeated_ground_truths,
                    completions=completions,
                    group_size=args.group_size,
                )
                write_jsonl(
                    run_dir / f"train_rollouts_step_{step:04d}.jsonl",
                    training_records,
                )
                log_rollouts_to_wandb(
                    wandb_run,
                    split="train",
                    step=step,
                    records=training_records,
                )

            should_validate = (
                step % args.validation_interval == 0
                or step == args.num_rollout_steps
            )
            if should_validate:
                server.sync_policy_weights(policy)
                validation_metrics, validation_records = evaluate(
                    server=server,
                    tokenizer=tokenizer,
                    examples=validation_examples,
                    prompt_template=prompt_template,
                    temperature=args.sampling_temperature,
                    top_p=args.sampling_top_p,
                    max_tokens=args.sampling_max_tokens,
                    seed=args.seed,
                    request_batch_size=args.vllm_request_batch_size,
                )
                log_metrics(
                    metrics_path,
                    wandb_run,
                    step=step,
                    metrics=validation_metrics,
                )
                write_jsonl(
                    run_dir / f"validation_rollouts_step_{step:04d}.jsonl",
                    validation_records,
                )
                log_rollouts_to_wandb(
                    wandb_run,
                    split="val",
                    step=step,
                    records=validation_records,
                )

        if args.save_final_model:
            model_output_dir = run_dir / "final_model"
            policy.save_pretrained(model_output_dir)
            tokenizer.save_pretrained(model_output_dir)
        write_run_status(
            run_dir,
            status="completed",
            last_completed_step=last_completed_step,
        )
    except GracefulShutdown as error:
        interrupted_signal = error.signum
        LOGGER.warning(
            "Received %s after step %d; shutting down.",
            signal.Signals(error.signum).name,
            last_completed_step,
        )
        write_run_status(
            run_dir,
            status="interrupted",
            last_completed_step=last_completed_step,
            error=signal.Signals(error.signum).name,
        )
    except Exception as error:
        write_run_status(
            run_dir,
            status="failed",
            last_completed_step=last_completed_step,
            error=repr(error),
        )
        raise
    finally:
        try:
            server.stop()
        finally:
            if wandb_run is not None:
                wandb_run.finish()

    if interrupted_signal is not None:
        raise SystemExit(128 + interrupted_signal)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )
    main()
