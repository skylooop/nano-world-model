"""Distributed training utilities for rank-aware logging and operations."""

import logging
import os
import sys

from colorama import Fore, Style, init as colorama_init
from pytorch_lightning.utilities.rank_zero import rank_zero_only

colorama_init(autoreset=True)


def get_rank() -> int:
    """Best-effort current distributed rank lookup."""
    for key in ("RANK", "LOCAL_RANK", "SLURM_PROCID"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                continue
    return 0


def rank_zero_only_bool() -> bool:
    return get_rank() == 0


# Check if current process is rank 0
is_rank_zero = rank_zero_only_bool()


class ColorFormatter(logging.Formatter):
    """Colored formatter for rank-0 console logs."""

    COLORS = {
        logging.DEBUG: Fore.CYAN,
        logging.INFO: Fore.GREEN,
        logging.WARNING: Fore.YELLOW,
        logging.ERROR: Fore.RED,
        logging.CRITICAL: Fore.MAGENTA,
    }

    def format(self, record):
        message = super().format(record)
        color = self.COLORS.get(record.levelno, "")
        prefix = f"{Fore.BLUE}[rank0]{Style.RESET_ALL} " if rank_zero_only_bool() else ""
        return f"{prefix}{color}{message}{Style.RESET_ALL}"


def get_rank_zero_logger(name: str, log_file: str | None = None) -> logging.Logger:
    """Create or retrieve a logger that only emits to console on rank 0."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if getattr(logger, "_nanowm_configured", False):
        return logger

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    if rank_zero_only_bool():
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(ColorFormatter("%(message)s"))
        logger.addHandler(console_handler)

    if log_file is not None:
        file_handler = logging.FileHandler(log_file, mode="a")
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter("[%(asctime)s][%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
        logger.addHandler(file_handler)

    logger._nanowm_configured = True
    return logger


def rank_zero_print(*args, **kwargs):
    """Print function that only executes on rank 0."""
    if rank_zero_only_bool():
        print(*args, **kwargs)


def rank_zero_log(func):
    """Decorator to ensure logging functions only execute on rank 0."""
    return rank_zero_only(func)
