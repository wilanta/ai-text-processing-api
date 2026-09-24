# Configure the root logger once at import time.
# The structured format (timestamp | level | logger-name | message) makes log
# lines easy to grep and parse by external log-aggregation tools.
import logging

logging.basicConfig(
    level=logging.INFO,
    format=("%(asctime)s | %(levelname)s | %(name)s | %(message)s"),
)

# All modules in this project import this singleton instead of creating their
# own logger, ensuring consistent naming and formatting across the codebase.
logger = logging.getLogger("ai-text-api")
