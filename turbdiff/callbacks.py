from typing import Any

from pytorch_lightning import Callback
from omegaconf import DictConfig, OmegaConf


class ConfigInCheckpoint(Callback):
    """Save the config in the checkpoint."""

    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict[str, Any]):
        checkpoint["config"] = OmegaConf.to_container(self.config, resolve=True)
