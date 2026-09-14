"""Plot GRPO metrics grouped by prompt style."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from plot_grpo_runs import (
    BOTTOM,
    DEFAULT_METRICS,
    HEIGHT,
    LEFT,
    RIGHT,
    TOP,
    WIDTH,
    _coordinate,
    _points,
    aggregate_by_step,
    load_metrics,
)


GROUP_COLORS = {
    "r1_zero": "#4c78a8",
    "question_only": "#f58518",
    "r1_zero_three_shot": "#54a24b",
}


def prompt_style_for_run(run_dir: Path) -> str:
    """Read the prompt style, including compatibility with older run configs."""
    config = json.loads((run_dir / "config.json").read_text())
    if config.get("prompt_style") is not None:
        return str(config["prompt_style"])

    prompt_filename = Path(config["prompt_path"]).name
    prompt_style_by_filename = {
        "question_only.prompt": "question_only",
        "r1_zero.prompt": "r1_zero",
        "r1_zero_three_shot_gsm8k.prompt": "r1_zero_three_shot",
    }
    try:
        return prompt_style_by_filename[prompt_filename]
    except KeyError as error:
        raise ValueError(
            f"Cannot infer prompt style for {run_dir} from {prompt_filename}"
        ) from error


def plot_metric(
    metric_name: str,
    grouped_series: dict[str, list[list[tuple[int, float]]]],
    output_path: Path,
) -> None:
    aggregates = {
        group_name: aggregate_by_step(run_series)
        for group_name, run_series in grouped_series.items()
    }
    all_steps = [
        step
        for steps, _, _ in aggregates.values()
        for step in steps
    ]
    all_values = []
    for _, means, standard_deviations in aggregates.values():
        all_values.extend(
            mean - std
            for mean, std in zip(means, standard_deviations, strict=True)
        )
        all_values.extend(
            mean + std
            for mean, std in zip(means, standard_deviations, strict=True)
        )

    x_min, x_max = min(all_steps), max(all_steps)
    y_min, y_max = min(all_values), max(all_values)
    y_padding = max((y_max - y_min) * 0.08, 1e-9)
    y_min -= y_padding
    y_max += y_padding

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: sans-serif; fill: #222; }</style>',
        f'<text x="{WIDTH / 2}" y="30" text-anchor="middle" '
        f'font-size="20">{html.escape(metric_name)} by prompt style</text>',
    ]

    for tick in range(6):
        fraction = tick / 5
        x = LEFT + fraction * (WIDTH - LEFT - RIGHT)
        step = x_min + fraction * (x_max - x_min)
        y = HEIGHT - BOTTOM - fraction * (HEIGHT - TOP - BOTTOM)
        value = y_min + fraction * (y_max - y_min)
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{TOP}" x2="{x:.2f}" '
                f'y2="{HEIGHT - BOTTOM}" stroke="#eeeeee"/>',
                f'<text x="{x:.2f}" y="{HEIGHT - BOTTOM + 24}" '
                f'text-anchor="middle" font-size="12">{step:.0f}</text>',
                f'<line x1="{LEFT}" y1="{y:.2f}" x2="{WIDTH - RIGHT}" '
                f'y2="{y:.2f}" stroke="#eeeeee"/>',
                f'<text x="{LEFT - 12}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-size="12">{value:.4g}</text>',
            ]
        )

    for group_index, (group_name, (steps, means, standard_deviations)) in enumerate(
        aggregates.items()
    ):
        color = GROUP_COLORS.get(group_name, "#777777")
        lower = [
            mean - std
            for mean, std in zip(means, standard_deviations, strict=True)
        ]
        upper = [
            mean + std
            for mean, std in zip(means, standard_deviations, strict=True)
        ]
        band = list(zip(steps, upper, strict=True)) + list(
            reversed(list(zip(steps, lower, strict=True)))
        )
        mean_points = _points(
            list(zip(steps, means, strict=True)),
            x_min,
            x_max,
            y_min,
            y_max,
        )
        elements.extend(
            [
                f'<polygon points="{_points(band, x_min, x_max, y_min, y_max)}" '
                f'fill="{color}" opacity="0.13"/>',
                f'<polyline points="{mean_points}" '
                f'fill="none" stroke="{color}" stroke-width="3"/>',
                f'<text x="{LEFT + 12}" y="{TOP + 18 * group_index}" '
                f'font-size="12" fill="{color}">{html.escape(group_name)}</text>',
            ]
        )

    elements.extend(
        [
            f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" '
            f'y2="{HEIGHT - BOTTOM}" stroke="#222222"/>',
            f'<line x1="{LEFT}" y1="{HEIGHT - BOTTOM}" x2="{WIDTH - RIGHT}" '
            f'y2="{HEIGHT - BOTTOM}" stroke="#222222"/>',
            f'<text x="{WIDTH / 2}" y="{HEIGHT - 20}" text-anchor="middle" '
            'font-size="14">training step</text>',
            '</svg>',
        ]
    )
    output_path.write_text("\n".join(elements) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot aggregate GRPO metrics grouped by prompt style."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_group: dict[str, list[dict[str, list[tuple[int, float]]]]] = {}
    for run_dir in args.run_dirs:
        prompt_style = prompt_style_for_run(run_dir)
        metrics_by_group.setdefault(prompt_style, []).append(load_metrics(run_dir))

    for metric_name in DEFAULT_METRICS:
        grouped_series = {
            group_name: [metrics[metric_name] for metrics in group_metrics]
            for group_name, group_metrics in metrics_by_group.items()
        }
        filename = metric_name.replace("/", "_") + ".svg"
        plot_metric(metric_name, grouped_series, args.output_dir / filename)


if __name__ == "__main__":
    main()
