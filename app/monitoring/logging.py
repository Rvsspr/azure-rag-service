from __future__ import annotations
import json
import logging
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, Optional, Union, Callable, Iterable, Tuple

"""
app/monitoring/logging.py

Lightweight logging configuration and helpers for the app.
- Provides configure_logging(...) to setup root logging (console + optional file).
- JSONFormatter for structured logs (no extra dependencies).
- get_logger(name) wrapper.
- add_structured_context to create LoggerAdapter with extra context.
- log_exceptions decorator/contextmanager helper.

This file is designed to be imported early (e.g. in app startup) and is dependency-free
(built on stdlib only).
"""


import logging.handlers


DEFAULT_MAX_BYTES = 10 * 1024 * 1024  # 10 MiB
DEFAULT_BACKUP_COUNT = 5


class JSONFormatter(logging.Formatter):
    """
    A simple JSON formatter that serializes records into a single-line JSON object.
    Includes timestamp, level, logger name, message, module info, and any extra fields.
    """

    def __init__(self, timestamp_format: str = "%Y-%m-%dT%H:%M:%S%z") -> None:
        super().__init__()
        self.timestamp_format = timestamp_format

    def format(self, record: logging.LogRecord) -> str:
        record_message = record.getMessage()
        # Base payload
        payload: Dict[str, Any] = {
            "timestamp": datetime.utcfromtimestamp(record.created).strftime(
                self.timestamp_format
            ),
            "level": record.levelname,
            "logger": record.name,
            "message": record_message,
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
            "process": record.process,
            "thread": record.thread,
        }

        # Include exception info if present
        if record.exc_info:
            payload["exc_info"] = "".join(traceback.format_exception(*record.exc_info))

        # Collect extra fields (anything not in LogRecord default attributes)
        standard_attrs = set(
            [
                "name",
                "msg",
                "args",
                "levelname",
                "levelno",
                "pathname",
                "filename",
                "module",
                "exc_info",
                "exc_text",
                "stack_info",
                "lineno",
                "funcName",
                "created",
                "msecs",
                "relativeCreated",
                "thread",
                "threadName",
                "processName",
                "process",
                "message",
            ]
        )
        extras: Dict[str, Any] = {}
        for k, v in record.__dict__.items():
            if k not in standard_attrs:
                try:
                    json.dumps({k: v})  # test serializability
                    extras[k] = v
                except Exception:
                    extras[k] = repr(v)
        if extras:
            payload["extra"] = extras

        return json.dumps(payload, ensure_ascii=False)


class ConsoleFormatter(logging.Formatter):
    """
    Readable console formatter (single-line). Minimal ANSI coloring for levels.
    """

    COLORS = {
        "DEBUG": "\033[37m",  # white
        "INFO": "\033[36m",  # cyan
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[41m",  # red background
    }
    RESET = "\033[0m"

    def __init__(self, fmt: Optional[str] = None, use_color: Optional[bool] = None) -> None:
        # default format
        fmt = fmt or "%(asctime)s %(levelname)s %(name)s: %(message)s"
        super().__init__(fmt=fmt, datefmt="%Y-%m-%d %H:%M:%S")
        if use_color is None:
            # Enable color on terminals that support it
            use_color = sys.stderr.isatty()
        self.use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        levelname = record.levelname
        if self.use_color and levelname in self.COLORS:
            levelname_colored = f"{self.COLORS[levelname]}{levelname}{self.RESET}"
            record.levelname = levelname_colored
        return super().format(record)


def _resolve_level(level: Union[str, int]) -> int:
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        try:
            return logging._nameToLevel[level.upper()]
        except Exception:
            return logging.INFO
    return logging.INFO


def configure_logging(
    level: Union[str, int] = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    backup_count: int = DEFAULT_BACKUP_COUNT,
    json_format: bool = False,
    root_logger_name: str = "",
) -> None:
    """
    Configure root logging for the process.

    Args:
        level: logging level (name or integer).
        log_file: optional path to a rotating file handler.
        max_bytes, backup_count: rotation settings for file handler.
        json_format: emit JSON structured logs if True (console + file).
        root_logger_name: if provided, configures logger with this name instead of root.
    """
    resolved_level = _resolve_level(level)
    logger = logging.getLogger(root_logger_name)
    # Avoid reconfiguring multiple times in long-running processes
    logger.handlers.clear()
    logger.setLevel(resolved_level)
    logger.propagate = False

    # Console handler (stdout)
    console_handler = logging.StreamHandler(stream=sys.stdout)
    if json_format:
        console_handler.setFormatter(JSONFormatter())
    else:
        console_handler.setFormatter(ConsoleFormatter())
    console_handler.setLevel(resolved_level)
    logger.addHandler(console_handler)

    # Optional rotating file handler
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        file_handler = logging.handlers.RotatingFileHandler(
            filename=log_file, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        if json_format:
            file_handler.setFormatter(JSONFormatter())
        else:
            # File should not contain ANSI color codes
            file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"))
        file_handler.setLevel(resolved_level)
        logger.addHandler(file_handler)


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Shortcut to get a configured logger. Call configure_logging(...) early in app startup.
    """
    return logging.getLogger(name)


class StructuredLoggerAdapter(logging.LoggerAdapter):
    """
    LoggerAdapter that merges context dict into log records under 'extra' keys.
    """

    def process(self, msg: Any, kwargs: Dict[str, Any]) -> Tuple[Any, Dict[str, Any]]:
        extra = kwargs.get("extra", {})
        # Merge adapter's context into extra
        merged = {**self.extra, **extra}
        kwargs["extra"] = merged
        return msg, kwargs


def add_structured_context(logger: logging.Logger, **context: Any) -> StructuredLoggerAdapter:
    """
    Return a LoggerAdapter that injects structured context into subsequent log calls.

    Example:
        logger = get_logger(__name__)
        ctx_logger = add_structured_context(logger, request_id=req_id, user_id=user_id)
        ctx_logger.info("handled request")
    """
    return StructuredLoggerAdapter(logger, context)


def log_exceptions(logger: Optional[logging.Logger] = None) -> Callable:
    """
    Decorator to log unhandled exceptions from a function at ERROR level and re-raise.

    Usage:
        @log_exceptions(get_logger(__name__))
        def handler(...): ...
    """

    def decorator(func: Callable) -> Callable:
        func_logger = logger or get_logger(func.__module__)

        def wrapper(*args: Any, **kwargs: Any):
            try:
                return func(*args, **kwargs)
            except Exception:
                func_logger.exception("Unhandled exception in %s", func.__qualname__)
                raise

        wrapper.__name__ = getattr(func, "__name__", "wrapped")
        wrapper.__doc__ = getattr(func, "__doc__", "")
        return wrapper

    return decorator


# Convenience: configure logging on import if env var is present (allows simple behavior)
_env_configured = False
if not _env_configured:
    env_level = os.environ.get("APP_LOG_LEVEL")
    env_file = os.environ.get("APP_LOG_FILE")
    env_json = os.environ.get("APP_LOG_JSON", "").lower() in ("1", "true", "yes")
    if env_level or env_file or env_json:
        configure_logging(level=env_level or "INFO", log_file=env_file, json_format=env_json)
        _env_configured = True