"""Plot final GRPO validation reward for a learning-rate sweep."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


WIDTH = 900
HEIGHT = 560
LEFT = 90
RIGHT = 35
TOP = 55
BOTTOM = 75
COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756")


def load_final_reward(run_dir: Path) -> tuple[float, float]:
    """Return the configured learning rate and final validation reward."""
    config = json.loads((run_dir / "config.json").read_text())
    final_validation = None
    with (run_dir / "metrics.jsonl").open() as input_file:
        for line in input_file:
            record = json.loads(line)
            if "val/reward" in record:
                final_validation = record

    if final_validation is None:
        raise ValueError(f"No validation reward found in {run_dir}")
    if final_validation["step"] != config["num_rollout_steps"]:
        raise ValueError(
            f"{run_dir} ended at validation step {final_validation['step']}, "
            f"expected {config['num_rollout_steps']}"
        )
    return float(config["learning_rate"]), float(final_validation["val/reward"])


def y_coordinate(value: float, y_max: float) -> float:
    plot_height = HEIGHT - TOP - BOTTOM
    return HEIGHT - BOTTOM - value / y_max * plot_height


def write_plot(
    results: dict[float, list[tuple[str, float]]],
    output_path: Path,
) -> None:
    learning_rates = sorted(results)
    means = [statistics.fmean(value for _, value in results[lr]) for lr in learning_rates]
    standard_deviations = [
        statistics.stdev(value for _, value in results[lr])
        if len(results[lr]) > 1
        else 0.0
        for lr in learning_rates
    ]
    y_max = max(
        0.01,
        max(mean + std for mean, std in zip(means, standard_deviations, strict=True))
        * 1.15,
    )
    plot_width = WIDTH - LEFT - RIGHT
    x_coordinates = [
        LEFT + index * plot_width / max(len(learning_rates) - 1, 1)
        for index in range(len(learning_rates))
    ]

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text { font-family: sans-serif; fill: #222; }</style>',
        f'<text x="{WIDTH / 2}" y="30" text-anchor="middle" font-size="20">'
        "Final validation reward by learning rate</text>",
    ]

    for tick in range(6):
        value = y_max * tick / 5
        y = y_coordinate(value, y_max)
        elements.extend(
            [
                f'<line x1="{LEFT}" y1="{y:.2f}" x2="{WIDTH - RIGHT}" '
                'y2="{y:.2f}" stroke="#eeeeee"/>',
                f'<text x="{LEFT - 12}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-size="12">{value:.0%}</text>',
            ]
        )

    mean_points = []
    for index, (learning_rate, x, mean, std) in enumerate(
        zip(
            learning_rates,
            x_coordinates,
            means,
            standard_deviations,
            strict=True,
        )
    ):
        mean_y = y_coordinate(mean, y_max)
        low_y = y_coordinate(max(0.0, mean - std), y_max)
        high_y = y_coordinate(min(y_max, mean + std), y_max)
        mean_points.append(f"{x:.2f},{mean_y:.2f}")
        elements.extend(
            [
                f'<line x1="{x:.2f}" y1="{high_y:.2f}" x2="{x:.2f}" '
                f'y2="{low_y:.2f}" stroke="#111111" stroke-width="2"/>',
                f'<line x1="{x - 7:.2f}" y1="{high_y:.2f}" x2="{x + 7:.2f}" '
                f'y2="{high_y:.2f}" stroke="#111111" stroke-width="2"/>',
                f'<line x1="{x - 7:.2f}" y1="{low_y:.2f}" x2="{x + 7:.2f}" '
                f'y2="{low_y:.2f}" stroke="#111111" stroke-width="2"/>',
                f'<text x="{x:.2f}" y="{HEIGHT - BOTTOM + 25}" '
                f'text-anchor="middle" font-size="13">{learning_rate:g}</text>',
                f'<text x="{x:.2f}" y="{mean_y - 14:.2f}" '
                f'text-anchor="middle" font-size="12">{mean:.1%}</text>',
            ]
        )
        points = results[learning_rate]
        for point_index, (run_name, reward) in enumerate(points):
            offset = (point_index - (len(points) - 1) / 2) * 12
            color = COLORS[point_index % len(COLORS)]
            elements.append(
                f'<circle cx="{x + offset:.2f}" '
                f'cy="{y_coordinate(reward, y_max):.2f}" r="5" '
                f'fill="{color}"><title>{html.escape(run_name)}: '
                f'{reward:.2%}</title></circle>'
            )

    elements.extend(
        [
            f'<polyline points="{" ".join(mean_points)}" fill="none" '
            'stroke="#111111" stroke-width="2"/>',
            f'<line x1="{LEFT}" y1="{TOP}" x2="{LEFT}" '
            f'y2="{HEIGHT - BOTTOM}" stroke="#222222"/>',
            f'<line x1="{LEFT}" y1="{HEIGHT - BOTTOM}" x2="{WIDTH - RIGHT}" '
            f'y2="{HEIGHT - BOTTOM}" stroke="#222222"/>',
            f'<text x="{WIDTH / 2}" y="{HEIGHT - 20}" text-anchor="middle" '
            'font-size="14">learning rate</text>',
            f'<text x="20" y="{HEIGHT / 2}" text-anchor="middle" font-size="14" '
            f'transform="rotate(-90 20 {HEIGHT / 2})">validation reward</text>',
            '</svg>',
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot final validation reward for GRPO learning-rate runs."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results: dict[float, list[tuple[str, float]]] = defaultdict(list)
    for run_dir in args.run_dirs:
        learning_rate, reward = load_final_reward(run_dir)
        results[learning_rate].append((run_dir.name, reward))

    write_plot(dict(results), args.output)
    for learning_rate in sorted(results):
        rewards = [reward for _, reward in results[learning_rate]]
        std = statistics.stdev(rewards) if len(rewards) > 1 else 0.0
        print(
            f"{learning_rate:g}\t{statistics.fmean(rewards):.8f}\t"
            f"{std:.8f}\t{len(rewards)}"
        )


if __name__ == "__main__":
    main()
