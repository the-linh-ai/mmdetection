# Copyright (c) OpenMMLab. All rights reserved.
import logging

from tqdm import tqdm
from mmcv.utils import get_logger

from .comm import is_main_process


def get_root_logger(log_file=None, log_level=logging.INFO):
    """Get root logger.

    Args:
        log_file (str, optional): File path of log. Defaults to None.
        log_level (int, optional): The level of logger.
            Defaults to logging.INFO.

    Returns:
        :obj:`logging.Logger`: The obtained logger
    """
    logger = get_logger(name='mmdet', log_file=log_file, log_level=log_level)

    return logger


class ProgressBar:
    """
    Extends tqdm's progress bar by handling multi-process scenarios.
    This progress bar only supports manually updating.
    """
    current_prog_bar = None

    def __init__(self, iterable, mininterval=1, **kwargs):
        # Destroy previously-initialized progress bar if possible
        if ProgressBar.current_prog_bar is not None:
            ProgressBar.current_prog_bar.close()
            ProgressBar.current_prog_bar = None

        # Only enable progress bar in the main process
        if is_main_process():
            if isinstance(iterable, int):
                total = iterable
                iterable = None
            else:
                total = None
            self.prog_bar = tqdm(
                iterable=iterable,
                total=total,
                mininterval=mininterval,
                **kwargs,
            )
            ProgressBar.current_prog_bar = self.prog_bar

        else:
            self.prog_bar = None

    def update(self, *args, **kwargs):
        if self.prog_bar is not None:
            self.prog_bar.update(*args, **kwargs)

    def close(self):
        if self.prog_bar is not None:
            self.prog_bar.close()
        ProgressBar.current_prog_bar = None
