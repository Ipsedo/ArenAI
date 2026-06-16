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

    for col in columns:
        match = regex_metric_name.match(col)
        if match is not None:
            name = match.group(1)
            metric_names.setdefault(name, [])

            metric_names[name].append(col)
        else:
            metric_names[col] = [col]

    fig, axes = plt.subplots(
        len(metric_names),
        1,
        figsize=(12, 3 * len(metric_names)),
        constrained_layout=True,
    )

    for i, (metric_group, metrics) in enumerate(metric_names.items()):
        ax = axes[i]

        ax.plot(metrics_df[metrics])

        ax.legend(metrics)
        ax.set_title(metric_group)

    fig.savefig(output_png_file_path)

    print(f'Metrics saved in "{output_png_file_path}"')
