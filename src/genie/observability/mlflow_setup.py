"""One-time MLflow tracking + LangChain autolog setup.

``init_mlflow`` is called at app startup. It degrades gracefully: when no
``MLFLOW_TRACKING_URI`` is configured MLflow runs in no-op mode (a warning, not a
crash), so the platform stays runnable without an MLflow server.
"""
import logging

import mlflow

from genie.platform.config import get_settings

_initialized = False
_log = logging.getLogger(__name__)


def init_mlflow(experiment_name: str | None = None) -> bool:
    """Configure MLflow tracking + LangChain autologging. Idempotent.

    Returns True if MLflow was configured against a tracking URI, False if no
    tracking URI was provided (in which case MLflow is in no-op mode and we
    log a warning instead of crashing).
    """
    global _initialized
    if _initialized:
        return True

    tracking_uri = get_settings().mlflow_tracking_uri
    experiment = experiment_name or get_settings().mlflow_experiment_name

    if not tracking_uri:
        _log.warning(
            "mlflow.tracking_uri_missing",
            extra={"attrs": {"hint": "set MLFLOW_TRACKING_URI to enable tracing"}},
        )
        _initialized = True
        return False

    try:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.langchain.autolog(log_traces=True, silent=True)
        _log.info(
            "mlflow.initialized",
            extra={"attrs": {"tracking_uri": tracking_uri, "experiment": experiment}},
        )
        _initialized = True
        return True
    except Exception as e:
        _log.error(
            "mlflow.init_failed",
            extra={"attrs": {"error": str(e)}},
            exc_info=True,
        )
        _initialized = True
        return False
