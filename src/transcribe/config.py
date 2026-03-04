import logging
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

if "GEMINI_API_KEY" in os.environ and "GOOGLE_API_KEY" not in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

def setup_logging(level=logging.INFO, log_file=None):
    """Set up the primary logger for the transcribe module."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
        
    # Configure the 'transcribe' logger specifically
    root_logger = logging.getLogger("transcribe")
    root_logger.setLevel(level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
        
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    for h in handlers:
        h.setFormatter(formatter)
        root_logger.addHandler(h)
    # Silence chatty libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("transcribe")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gemma-3-4b-it")
