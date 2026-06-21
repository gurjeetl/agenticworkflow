"""Structured (JSON) logging wired to also surface logs inside MLflow spans.

``configure_logging`` installs two root handlers: a JSON stdout formatter for
machine-readable logs, and :class:`MLflowSpanHandler` which attaches each record
to the active MLflow span so logs and traces stay correlated. Modules obtain a
logger via :func:`get_logger` and pass structured fields through ``extra={"attrs": ...}``.
"""
import json
import logging
import sys
from datetime import datetime, timezone

import mlflow

_configured = False


class JsonFormatter(logging.Formatter):
    """Render each log record as a single JSON line, flattening ``extra['attrs']``."""

    def format(self, record: logging.LogRecord) -> str:
        """Build the JSON payload (time/level/logger/msg + attrs + any exception)."""
        payload = {
            "time": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        attrs = getattr(record, "attrs", None)
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if k not in payload:
                    payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


class MLflowSpanHandler(logging.Handler):
    """Attach every log record emitted inside an active MLflow span as a span event."""

    def emit(self, record: logging.LogRecord) -> None:
        """Add the record as a span event; silently no-op when no span is active."""
        try:
            span = mlflow.get_current_active_span()
            if span is None:
                return
            attrs = {"logger": record.name, "level": record.levelname}
            extra_attrs = getattr(record, "attrs", None)
            if isinstance(extra_attrs, dict):
                for k, v in extra_attrs.items():
                    attrs[k] = v if isinstance(v, (str, int, float, bool)) else str(v)
            if record.exc_info:
                attrs["exception"] = self.format(record)
            span.add_event(f"{record.levelname}: {record.getMessage()}", attributes=attrs)
        except Exception:
            pass


def configure_logging(level: str | None = None) -> None:
    """Install the JSON + MLflow-span root handlers once (idempotent)."""
    global _configured
    if _configured:
        return
    log_level = getattr(logging, (level or "INFO").upper(), logging.INFO)

    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(JsonFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(stream)
    root.addHandler(MLflowSpanHandler())
    root.setLevel(log_level)
    _configured = True


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a standard logger (typically ``get_logger(__name__)``)."""
    return logging.getLogger(name)
