import logging
from pathlib import Path

import uvicorn

LOG_DIR = Path.home() / ".openclaw" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "bobvoice.log"


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_FILE, mode="a"),
        ],
    )
    logging.getLogger("bobvoice").info("Bob Voice starting on 0.0.0.0:8421")
    uvicorn.run(
        "bobvoice.main:app",
        host="0.0.0.0",
        port=8421,
        reload=False,
        log_config=None,
    )


if __name__ == "__main__":
    main()
