"""Plot GRPO metrics across repeated random-seed runs as SVG files."""

from __future__ import annotations

import argparse
import html
import json
import statistics
from collections import defaultdict
from pathlib import Path


DEFAULT_METRICS = (
    "train/loss",
    "train/gradient_norm",
    "train/token_entropy",
    "train/reward",
    "train/format_reward",
    "val/reward",
    "val/format_reward",
    "val/average_response_length",
)

WIDTH = 960
HEIGHT = 600
LEFT = 90
RIGHT = 30
TOP = 55
BOTTOM = 70
COLORS = ("#4c78a8", "#f58518", "#54a24b", "#e45756")


def load_metrics(run_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Load one run's JSONL metrics, grouped by metric name."""
    metrics_path = run_dir / "metrics.jsonl"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"Missing metrics file: {metrics_path}")

    series: dict[str, list[tuple[int, float]]] = defaultdict(list)
    with metrics_path.open() as input_file:
        for line in input_file:
            record = json.loads(line)
            step = int(record["step"])
            for name, value in record.items():
                if name != "step":
                    series[name].append((step, float(value)))
    return dict(series)


def aggregate_by_step(
    run_series: list[list[tuple[int, float]]],
) -> tuple[list[int], list[float], list[float]]:
    """Compute the mean and sample standard deviation at every shared step."""
    values_by_step: dict[int, list[float]] = defaultdict(list)
    for series in run_series:
        for step, value in series:
            values_by_step[step].append(value)

    num_runs = len(run_series)
    shared_steps = sorted(
        step for step, values in values_by_step.items() if len(values) == num_runs
    )
    means = [statistics.fmean(values_by_step[step]) for step in shared_steps]
    standard_deviations = [
        statistics.stdev(values_by_step[step]) if num_runs > 1 else 0.0
        for step in shared_steps
    ]
    return shared_steps, means, standard_deviations


def _coordinate(
    value: float,
    low: float,
    high: float,
    start: float,
    end: float,
) -> float:
    if high == low:
        return (start + end) / 2
    return start + (value - low) / (high - low) * (end - start)


def _points(
    series: list[tuple[int, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> str:
    points = []
    for step, value in series:
        x = _coordinate(step, x_min, x_max, LEFT, WIDTH - RIGHT)
        y = _coordinate(value, y_min, y_max, HEIGHT - BOTTOM, TOP)
        points.append(f"{x:.2f},{y:.2f}")
    return " ".join(points)


def plot_metric(
    metric_name: str,
    run_names: list[str],
    run_series: list[list[tuple[int, float]]],
    output_path: Path,
) -> None:
    """Plot individual runs and their mean plus or minus one standard deviation."""
    steps, means, standard_deviations = aggregate_by_step(run_series)
    lower = [mean - std for mean, std in zip(means, standard_deviations, strict=True)]
    upper = [mean + std for mean, std in zip(means, standard_deviations, strict=True)]

    all_steps = [step for series in run_series for step, _ in series]
    all_values = [value for series in run_series for _, value in series]
    all_values.extend(lower)
    all_values.extend(upper)
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
        f'font-size="20">{html.escape(metric_name)}</text>',
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

    upper_points = list(zip(steps, upper, strict=True))
    lower_points = list(reversed(list(zip(steps, lower, strict=True))))
    band_points = _points(
        upper_points + lower_points,
        x_min,
        x_max,
        y_min,
        y_max,
    )
    elements.append(
        f'<polygon points="{band_points}" fill="#222222" opacity="0.14"/>'
    )

    for index, (run_name, series) in enumerate(
        zip(run_names, run_series, strict=True)
    ):
        color = COLORS[index % len(COLORS)]
        elements.append(
            f'<polyline points="{_points(series, x_min, x_max, y_min, y_max)}" '
            f'fill="none" stroke="{color}" stroke-width="1.5" opacity="0.55"/>'
        )
        legend_y = TOP + 18 * index
        elements.append(
            f'<text x="{LEFT + 12}" y="{legend_y}" font-size="12" '
            f'fill="{color}">{html.escape(run_name)}</text>'
        )

    mean_series = list(zip(steps, means, strict=True))
    elements.extend(
        [
            f'<polyline points="{_points(mean_series, x_min, x_max, y_min, y_max)}" '
            'fill="none" stroke="#111111" stroke-width="3"/>',
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
        description="Plot individual and aggregate metrics from GRPO runs."
    )
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = [load_metrics(run_dir) for run_dir in args.run_dirs]
    run_names = [run_dir.name for run_dir in args.run_dirs]
    for metric_name in DEFAULT_METRICS:
        run_series = [metrics.get(metric_name, []) for metrics in all_metrics]
        if any(not series for series in run_series):
            missing_runs = [
                run_name
                for run_name, series in zip(run_names, run_series, strict=True)
                if not series
            ]
            raise ValueError(f"{metric_name} is missing from runs: {missing_runs}")

        filename = metric_name.replace("/", "_") + ".svg"
        plot_metric(
            metric_name=metric_name,
            run_names=run_names,
            run_series=run_series,
            output_path=args.output_dir / filename,
        )


if __name__ == "__main__":
    main()
