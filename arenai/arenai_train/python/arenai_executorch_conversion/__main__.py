import argparse
import re
from os import makedirs
from os.path import dirname, exists, isdir

import torch as th
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

from .constants import (
    ENEMY_NB_ACTIONS,
    ENEMY_PROPRIOCEPTION_SIZE,
    ENEMY_VISION_SIZE,
)
from .loader import load_neutral_state_into
from .model import SacActor


def _channels(string: str) -> list[tuple[int, int]]:
    regex_match = re.compile(
        r"^ *\[(?: *\( *\d+ *, *\d+ *\) *,)* *\( *\d+ *, *\d+ *\) *] *$"
    )
    regex_layer = re.compile(r"\( *\d+ *, *\d+ *\)")
    regex_channel = re.compile(r"\d+")

    assert regex_match.match(string), "usage : [(10, 20), (20, 40), ...]"

    def _match_channels(layer_str: str) -> tuple[int, int]:
        matched = regex_channel.findall(layer_str)
        assert len(matched) == 2
        return int(matched[0]), int(matched[1])

    return [_match_channels(layer) for layer in regex_layer.findall(string)]


def _groups(string: str) -> list[int]:
    regex_match = re.compile(r"^ *\[(?: *\d+ *,)* *\d+ *] *$")
    regex_group = re.compile(r"\d+")

    assert regex_match.match(string), "usage : [4, 8, 16, ...]"

    return [int(group) for group in regex_group.findall(string)]


def main() -> None:
    parser = argparse.ArgumentParser("arenai convert model to PTE")

    parser.add_argument("-o", "--output_pte", type=str, required=True)
    parser.add_argument(
        "-i", "--input_state_dict_folder", type=str, required=True
    )

    parser.add_argument("--sensors_hidden_size", type=int, default=256)
    parser.add_argument("--actor_hidden_size", type=int, default=1536)
    parser.add_argument(
        "--group_norm_nums", type=_groups, default=[2, 4, 8, 16, 32, 64, 64]
    )
    parser.add_argument(
        "--vision_channels",
        type=_channels,
        default=[
            (3, 8),
            (8, 16),
            (16, 32),
            (32, 64),
            (64, 128),
            (128, 256),
            (256, 512),
        ],
    )

    args = parser.parse_args()

    if not exists(args.input_state_dict_folder) or not isdir(
        args.input_state_dict_folder
    ):
        raise NotADirectoryError(
            f'"{args.input_state_dict_folder}" does not exist or is not a directory'
        )

    output_folder = dirname(args.output_pte)
    if not exists(output_folder):
        makedirs(output_folder, exist_ok=True)

    with th.no_grad():

        actor = SacActor(
            ENEMY_PROPRIOCEPTION_SIZE,
            ENEMY_NB_ACTIONS,
            args.sensors_hidden_size,
            args.actor_hidden_size,
            args.vision_channels,
            args.group_norm_nums,
        )

        load_neutral_state_into(actor, args.input_state_dict_folder)

        print("Model loaded from C++ !")

        batch = Dim("batch", min=1, max=8)
        dynamic_shapes = (
            {0: batch},
            {0: batch},
        )

        example_input = (
            th.randn(2, 3, ENEMY_VISION_SIZE, ENEMY_VISION_SIZE),
            th.randn(2, ENEMY_PROPRIOCEPTION_SIZE),
        )

        exported_program = export(
            actor, example_input, dynamic_shapes=dynamic_shapes
        )

        executorch_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        with open(args.output_pte, "wb") as file:
            file.write(executorch_program.buffer)

        print("Model saved at", args.output_pte)


if __name__ == "__main__":
    main()
