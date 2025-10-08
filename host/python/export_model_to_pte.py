import torch as th
from torch.export import export, Dim
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower
from python.model import SacActor
from python.loader import load_neutral_state_into


def main() -> None:
    output_pte = "/home/samuel/Téléchargements/actor.pte"
    state_dict_path = "/home/samuel/Téléchargements/actor_export"
    nb_sensors = 3 * 3 + 3 * 3
    nb_actions = 2 + 2 + 3

    actor = SacActor(nb_sensors, nb_actions, 64, 256)

    load_neutral_state_into(actor, state_dict_path)

    batch = Dim("batch", min=1, max=32)
    dynamic_shapes = (
        {0: batch},
        {0: batch},
    )

    example_input = (th.randn(2, 3, 128, 128), th.randn(2, nb_sensors))

    exported_program = export(actor, example_input, dynamic_shapes=dynamic_shapes)

    gm = exported_program.module()
    with th.no_grad():
        print(gm(*example_input))

    executorch_program = to_edge_transform_and_lower(
        exported_program,
        partitioner=[XnnpackPartitioner()],
    ).to_executorch()

    with open(output_pte, "wb") as file:
        file.write(executorch_program.buffer)

    with th.no_grad():
        print(actor(*example_input))


if __name__ == '__main__':
    main()
