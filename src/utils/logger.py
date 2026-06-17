import json
import os
from tempfile import mkdtemp

from omegaconf import OmegaConf
import wandb


class Logger:
    def __init__(
        self,
        output_dir=None,
        config={},
        use_wandb=False,
        wandb_init_kw={},
        overwrite=False,
        console=False,
    ):

        if not output_dir:
            output_dir = mkdtemp()
        if os.path.exists(output_dir) and overwrite:
            print(f"Overwriting existing output directory at {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
        elif os.path.exists(output_dir) and not overwrite:
            raise FileExistsError(
                f"Output directory already exists at {output_dir} and overwrite is False"
            )
        else:
            os.makedirs(output_dir, exist_ok=True)
        self.output_dir = output_dir
        self.config = config

        if not isinstance(config, OmegaConf):
            config = OmegaConf.create(config)

        OmegaConf.save(
            config, os.path.join(output_dir, "config-resolved.yaml"), resolve=True
        )
        OmegaConf.save(config, os.path.join(output_dir, "config.yaml"), resolve=False)

        self.use_wandb = use_wandb
        if self.use_wandb:
            wandb.init(**wandb_init_kw, config=OmegaConf.to_object(config))
            wandb.save(os.path.join(self.output_dir, "config-resolved.yaml"))
            wandb.save(os.path.join(self.output_dir, "config.yaml"))

        self.metrics_jsonl = open(os.path.join(self.output_dir, "metrics.jsonl"), "w")
        self.print = console

    def log_scalars(self, data):
        if self.use_wandb:
            wandb.log(data)

        self.metrics_jsonl.write(json.dumps(data) + "\n")
        self.metrics_jsonl.flush()

        if self.print:
            print(data)

    def __call__(self, data):
        self.log_scalars(data)
