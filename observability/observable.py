from __future__ import annotations

import asyncio
import concurrent.futures
import logging
from contextlib import contextmanager
from functools import wraps
from typing import Any, Callable, Coroutine

import mlflow
from mlflow.entities import SpanType

_MAX_VALUE_LEN = 500
_STATE_KEYS = ("user_input", "thread_id", "intent", "location", "iteration_count",
               "active_agent", "next_action", "is_complete", "error")


def _safe_repr(value: Any, max_len: int = _MAX_VALUE_LEN) -> str:
    try:
        s = value if isinstance(value, str) else repr(value)
    except Exception:
        return "<unreprable>"
    return s if len(s) <= max_len else s[:max_len] + "...[truncated]"


def _stringify_attrs(attrs: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None:
            continue
        if isinstance(v, (str, int, float, bool)):
            out[k] = v
        else:
            out[k] = _safe_repr(v)
    return out


def _capture_inputs(args: tuple, kwargs: dict) -> dict[str, Any]:
    if len(args) == 1 and isinstance(args[0], dict):
        state = args[0]
        captured = {k: state[k] for k in _STATE_KEYS if k in state and state[k] not in (None, "")}
        if captured:
            return _stringify_attrs(captured)
    payload: dict[str, Any] = {}
    if args:
        payload["args"] = _safe_repr(args)
    if kwargs:
        payload["kwargs"] = _safe_repr(kwargs)
    return payload


def _capture_outputs(result: Any) -> dict[str, Any]:
    if isinstance(result, dict):
        captured = {k: result[k] for k in _STATE_KEYS if k in result and result[k] not in (None, "")}
        if captured:
            return _stringify_attrs(captured)
        return {}
    return {"result": _safe_repr(result)}


def _wrap_with_mlflow_span(method: Callable, owner_cls: type) -> Callable:
    if getattr(method, "_mlflow_traced", False):
        return method
    fallback_kind = getattr(owner_cls, "_component_kind", "component")
    fallback_span_type = getattr(owner_cls, "_span_type", SpanType.CHAIN)
    method_name = method.__name__

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        cls = type(self)
        component_kind = getattr(cls, "_component_kind", fallback_kind)
        span_type = getattr(cls, "_span_type", fallback_span_type)
        span_name = f"{component_kind}.{cls.__name__}.{method_name}"
        with mlflow.start_span(name=span_name, span_type=span_type) as span:
            try:
                inputs = _capture_inputs(args, kwargs)
                if inputs:
                    span.set_inputs(inputs)
                span.set_attribute("component.kind", component_kind)
                span.set_attribute("component.class", cls.__name__)
            except Exception:
                pass
            try:
                result = method(self, *args, **kwargs)
            except Exception as e:
                try:
                    span.record_exception(e)
                except Exception:
                    pass
                raise
            try:
                outputs = _capture_outputs(result)
                if outputs:
                    span.set_outputs(outputs)
            except Exception:
                pass
            return result

    wrapper._mlflow_traced = True
    return wrapper


class Observable:
    """Inherit to gain MLflow tracing + correlated logging.

    Subclasses declare which methods to auto-wrap in `_traced_methods` and the
    semantic span type via `_component_kind` and `_span_type`. Works for agents,
    tools, orchestrators, memory layers, API handlers — anything in the platform.
    """

    _traced_methods: tuple[str, ...] = ()
    _component_kind: str = "component"
    _span_type: str = SpanType.CHAIN

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        for name in cls._traced_methods:
            method = cls.__dict__.get(name) or getattr(cls, name, None)
            if not callable(method) or getattr(method, "_mlflow_traced", False):
                continue
            setattr(cls, name, _wrap_with_mlflow_span(method, cls))

    # --- Instance API ---

    def log_event(self, name: str, **attrs: Any) -> None:
        span = mlflow.get_current_active_span()
        if span is None:
            return
        try:
            span.add_event(name, attributes=_stringify_attrs(attrs))
        except Exception:
            pass

    def log_metric(self, name: str, value: float, step: int | None = None) -> None:
        try:
            mlflow.log_metric(name, value, step=step)
        except Exception:
            pass

    def log_param(self, name: str, value: Any) -> None:
        try:
            mlflow.log_param(name, value)
        except Exception:
            pass

    @contextmanager
    def span(self, name: str, span_type: str = SpanType.CHAIN):
        with mlflow.start_span(name=name, span_type=span_type) as s:
            yield s

    def log(self, level: str, event: str, **attrs: Any) -> None:
        logger = logging.getLogger(self.__class__.__module__)
        log_fn = getattr(logger, level.lower(), logger.info)
        exc_info = attrs.pop("exc_info", False)
        log_fn(event, extra={"attrs": _stringify_attrs(attrs)}, exc_info=exc_info)
        span = mlflow.get_current_active_span()
        if span is not None:
            try:
                span.add_event(f"{level.upper()}: {event}", attributes=_stringify_attrs(attrs))
            except Exception:
                pass

    def _run_async(self, coro: Coroutine):
        """Run an async coroutine from sync code, even if an event loop is already running.

        Inside a running loop (e.g. inside a FastAPI handler), runs the coroutine on a
        worker thread that owns its own loop. Outside any loop, uses asyncio.run directly.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(coro)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
            return ex.submit(asyncio.run, coro).result()
