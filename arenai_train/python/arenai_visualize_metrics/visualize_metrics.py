import re

import pandas as pd
from matplotlib import pyplot as plt


def metrics_csv_to_plot(
    csv_path: str, output_png_file_path: str, sep: str = ";"
) -> None:
    print(f'Will load "{csv_path}"')

    metrics_df = pd.read_csv(csv_path, sep=sep, header=0)
    metrics_df.set_index("index", inplace=True)

    columns = metrics_df.columns

    regex_metric_name = re.compile(r"^(.+)_([μσ])$")

    metric_names = {}

    raw_col_identifier = "raw"

    for col in columns:
        match = regex_metric_name.match(col)
        if match is not None:
            name = match.group(1)
            kind = match.group(2)
            metric_names.setdefault(name, {})

            metric_names[name][kind] = col
        else:
            metric_names[col] = {raw_col_identifier: col}

    fig, axes = plt.subplots(
        len(metric_names),
        1,
        figsize=(12, 3 * len(metric_names)),
        constrained_layout=True,
    )

    ticks = list(range(len(metrics_df)))

    for i, (metric_group, metrics_dict) in enumerate(metric_names.items()):
        ax = axes[i]

        if len(metrics_dict) == 1 and raw_col_identifier in metrics_dict:
            raw_col = metrics_dict[raw_col_identifier]
            ax.plot(metrics_df[raw_col])

            ax.legend(raw_col)
        else:
            mean_col = metrics_dict["μ"]
            std_col = metrics_dict["σ"]

            mean = metrics_df[mean_col]

            lower_bound = mean - metrics_df[std_col]
            upper_bound = mean + metrics_df[std_col]

            ax.plot(ticks, mean)
            ax.fill_between(ticks, lower_bound, upper_bound, alpha=0.2)

            ax.legend(["mean", "std"])

        ax.set_title(metric_group)

    fig.savefig(output_png_file_path)

    print(f'Metrics saved in "{output_png_file_path}"')
