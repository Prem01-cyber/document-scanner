import json
import logging
from datetime import datetime
from typing import Any, Dict


class JSONFormatter(logging.Formatter):
    """Simple JSON log formatter for structured logs."""

    def format(self, record: logging.LogRecord) -> str:
        log: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        if record.exc_info:
            log["exc_info"] = self.formatException(record.exc_info)

        # Include extra fields if present
        for key, value in record.__dict__.items():
            if key in ("args", "msg", "created", "levelname", "name", "exc_info", "exc_text", "msecs", "relativeCreated", "asctime"):
                continue
            if key.startswith("_"):
                continue
            try:
                json.dumps(value)
                log[key] = value
            except Exception:
                log[key] = str(value)

        return json.dumps(log, ensure_ascii=False)


def configure_json_logging(level: int = logging.INFO) -> None:
    """Configure root logger to emit JSON logs."""
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(level)


