import argparse

from .visualize_metrics import metrics_csv_to_plot


def main() -> None:
    parser = argparse.ArgumentParser("arenai_visualize_metrics")

    parser.add_argument(
        "csv_path",
        type=str,
        help="The result CSV file from arenai_agent C++ program",
    )
    parser.add_argument(
        "-o",
        "--output_png",
        type=str,
        default="arenai_agent_metrics.png",
        help="The output plot image of metrics",
    )

    args = parser.parse_args()

    metrics_csv_to_plot(args.csv_path, args.output_png)


if __name__ == "__main__":
    main()
