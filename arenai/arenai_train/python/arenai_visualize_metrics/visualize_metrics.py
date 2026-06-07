import re

import pandas as pd
from matplotlib import pyplot as plt


def metrics_csv_to_plot(
    csv_path: str, output_png_file_path: str, sep: str = ";"
) -> None:
    print(f'Will load "{csv_path}"')

    metrics_df = pd.read_csv(csv_path, sep=sep)
    metrics_df.set_index("index", inplace=True)

    columns = metrics_df.columns

    regex_metric_name = re.compile(r"^(.+)_[^_]+$")

    metric_names = []

    for col in columns:
        match = regex_metric_name.match(col)
        if match is not None:
            name = match.group(1)
            if name not in metric_names:
                metric_names.append(name)

    fig, axes = plt.subplots(
        len(metric_names),
        1,
        figsize=(12, 3 * len(metric_names)),
        constrained_layout=True,
    )

    for i, m in enumerate(metric_names):
        ax = axes[i]
        sub_columns = [f"{m}_mean", f"{m}_std"]

        ax.plot(metrics_df[sub_columns])

        ax.legend(sub_columns)
        ax.set_title(m)

    fig.savefig(output_png_file_path)

    print(f'Metrics saved in "{output_png_file_path}"')
