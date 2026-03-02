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
        
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers
    )
    # Silence chatty libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

logger = logging.getLogger("transcribe")
DEFAULT_MODEL_NAME = os.getenv("MODEL_NAME", "gemma-3-4b-it")
