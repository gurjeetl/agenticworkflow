import logging
import os

import mlflow

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

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    experiment = experiment_name or os.getenv("MLFLOW_EXPERIMENT_NAME", "base-agent-framework")

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
