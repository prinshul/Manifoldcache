from argparse import ArgumentParser, Namespace
import torch

from pathlib import Path
import sys
sys.path.append("..")
from diffcsp.common.data_utils import preprocess


def main(args: Namespace) -> None:
    csvs = [
        str(args.directory / "train.csv"),
        str(args.directory / "val.csv"),
        str(args.directory / "test.csv"),
    ]
    pts = [
        str(args.directory / "train_sym.pt"),
        str(args.directory / "val_sym.pt"),
        str(args.directory / "test_sym.pt"),
    ]

    for csv, pt in zip(csvs, pts):
        print("working on", csv)
        cached_data = preprocess(
            csv,
            num_workers=94,
            niggli=True,
            primitive=False,
            graph_method="crystalnn",
            # symprec=0.1,
            # angle_tolerance=5,
            prop_list=["formation_energy_per_atom"],
            use_space_group=True,
        )
        torch.save(cached_data, pt)
        print("done with", csv)


if __name__ == "__main__":
    parser = ArgumentParser(description="Process crystal data")
    parser.add_argument(
        "directory",
        type=Path,
        help="Directory containing the CSV files to process.",
    )
    args = parser.parse_args()

    main(args)
