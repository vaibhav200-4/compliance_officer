import sys
from pathlib import Path

from loguru import logger


# Create logs directory if it doesn't exist
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


# Remove default logger
logger.remove()


# Console Logger
logger.add(
    sys.stdout,
    level="INFO",
    colorize=True,
    format=(
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    ),
)


# File logging must not prevent ingestion when the log directory is read-only.
try:
    logger.add(
        LOG_DIR / "app.log",
        rotation="10 MB",
        retention="10 days",
        compression="zip",
        level="DEBUG",
        enqueue=True,
        backtrace=True,
        diagnose=True,
        format=(
            "{time:YYYY-MM-DD HH:mm:ss} | "
            "{level} | "
            "{name}:{function}:{line} | "
            "{message}"
        ),
    )
except PermissionError:
    logger.warning("File logging is unavailable; continuing with console logging only.")


def get_logger():
    """
    Returns configured logger instance.
    """
    return logger
