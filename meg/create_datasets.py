from datasets.datasets import MegDataset
import hydra

from pathlib import Path
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="config")
def main(cfg: DictConfig):
    input_dir = Path(cfg.build.input_directory) / cfg.data.dataset
    output_dir = Path(cfg.build.output_directory)

    name = cfg.data.dataset

    branch = cfg.build.branch
    if branch:
        relative_path = Path(name, branch)
    else:
        relative_path = Path(name, "dev")

    output_dir = output_dir / relative_path

    ds = MegDataset(
        input_dir=input_dir,
        output_dir=output_dir,
        test_run=cfg.build.test_run,
        **cfg.data,
    )
    ds.generate()
    if not cfg.build.test_run:
        with (output_dir / "build").open("w") as f:
            f.write(str(cfg.build.id))


if __name__ == "__main__":
    main()
