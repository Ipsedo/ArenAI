import argparse

import torch as th
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower
from torch.export import Dim, export

from python.loader import load_neutral_state_into
from python.model import SacActor


def main() -> None:
    parser = argparse.ArgumentParser("phyvr convert model to PTE")

    parser.add_argument("-o", "--output-pte", type=str, required=True)
    parser.add_argument(
        "-i", "--input-state-dict-folder", type=str, required=True
    )

    parser.add_argument("--hidden-size-sensors", type=int, default=192)
    parser.add_argument("--hidden-size", type=int, default=1024)

    args = parser.parse_args()

    with th.no_grad():
        nb_sensors = (3 * 2 + 4 + 3) * (6 + 3)
        nb_actions = 2 + 2 + 1

        actor = SacActor(
            nb_sensors, nb_actions, args.hidden_size_sensors, args.hidden_size
        )

        load_neutral_state_into(actor, args.input_state_dict_folder)

        print("Model loaded from C++ !")

        batch = Dim("batch", min=1, max=32)
        dynamic_shapes = (
            {0: batch},
            {0: batch},
        )

        example_input = (th.randn(2, 3, 128, 128), th.randn(2, nb_sensors))

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
