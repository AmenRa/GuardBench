__all__ = [
    "benchmark_effectiveness",
    "benchmark_efficiency",
    "benchmark",
    "CustomDataset",
    "download_all",
    "Report",
]

import os
import sys
from functools import partialmethod
from pathlib import Path

from loguru import logger

from .benchmark.effectiveness import benchmark
from .benchmark.effectiveness import benchmark as benchmark_effectiveness
from .benchmark.efficiency import benchmark as benchmark_efficiency
from .datasets import download_all
from .datasets.custom_dataset import CustomDataset
from .report import Report

# Set environment variables ----------------------------------------------------
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Allow user to set a different path in .bash_profile
if "GUARDBENCH_BASE_PATH" not in os.environ:
    os.environ["GUARDBENCH_BASE_PATH"] = str(Path.home() / ".guardbench")


# Setup logger -----------------------------------------------------------------
logger.remove()
logger.add(
    sys.stdout,
    colorize=True,
    format="<green>{time:YYYY-MM-DD at HH:mm:ss}</> | <bold>GuardBench</> | <bold><level>{level: <8}</></> | <bold><level>{message}</></>",
)
logger.level("START", no=33, color="<blue>")
logger.__class__.start = partialmethod(logger.__class__.log, "START")
