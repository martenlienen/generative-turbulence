# SPDX-FileCopyrightText: © 2024 Marten Lienen <m.lienen@tum.de> & Technical University of Munich
#
# SPDX-License-Identifier: MIT

import time
from datetime import timedelta

import pytorch_lightning as pl
from pytimeparse import parse as timeparse

from .utils import get_logger

log = get_logger()


class TimeLimit(pl.Callback):
    def __init__(self, train_limit: str):
        super().__init__()

        self.start_time = None
        self.limit = timeparse(train_limit)

    def on_train_start(self, trainer, task):
        self.start_time = time.monotonic()

        delta = timedelta(seconds=self.limit)
        log.info(f"Training will be stopped in {delta}")

    def on_train_batch_end(self, trainer, task, outputs, batch, batch_idx):
        if self.start_time is None:
            return

        elapsed = time.monotonic() - self.start_time

        if elapsed > self.limit:
            # Ensure validation after stopping even if we did not stop in a "validation
            # epoch"
            trainer.check_val_every_n_epoch = None

            trainer.should_stop = True
            delta = timedelta(seconds=self.limit)
            log.info(f"Training stopped after {delta}")
