"""
Logging utilities.
"""

import logging
import sys
from pathlib import Path
from rich.logging import RichHandler
from rich.console import Console

def setup_logging(log_dir: Path = Path("logs")):
    """Configure structured logging."""
    log_dir.mkdir(exist_ok=True)
    
    # Console handler (rich)
    rich_handler = RichHandler(
        rich_tracebacks=True,
        show_level=True,
        show_path=True
    )
    
    # File handler
    file_handler = logging.FileHandler(log_dir / "extraction.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[rich_handler, file_handler]
    )
