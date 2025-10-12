import torch as th
from torch.export import export, Dim
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from python.model import SacActor
from python.loader import load_neutral_state_into


def main() -> None:
    with th.no_grad():
        output_pte = "/home/samuel/Téléchargements/actor.pte"
        state_dict_path = "/home/samuel/Téléchargements/actor_export"
        nb_sensors = (3 * 2 + 4 + 3) * (6 + 3)
        nb_actions = 2 + 2 + 1

        actor = SacActor(nb_sensors, nb_actions, 160, 320)

        load_neutral_state_into(actor, state_dict_path)

        print("Model loaded from C++ !")

        batch = Dim("batch", min=1, max=32)
        dynamic_shapes = (
            {0: batch},
            {0: batch},
        )

        example_input = (th.randn(2, 3, 128, 128), th.randn(2, nb_sensors))

        exported_program = export(actor, example_input, dynamic_shapes=dynamic_shapes)

        executorch_program = to_edge_transform_and_lower(
            exported_program,
            partitioner=[XnnpackPartitioner()],
        ).to_executorch()

        with open(output_pte, "wb") as file:
            file.write(executorch_program.buffer)

        print("Model saved at", output_pte)


if __name__ == '__main__':
    main()
