"""Tests for the logging setup."""

import logging

from src.logging_config import get_logger


def test_get_logger_returns_logger():
    assert isinstance(get_logger(), logging.Logger)


def test_logger_is_configured_once():
    # Idempotent: repeated calls return the same logger and don't stack handlers.
    first = get_logger()
    handler_count = len(first.handlers)
    second = get_logger()
    assert first is second
    assert len(second.handlers) == handler_count


def test_logger_has_console_and_file_handlers():
    logger = get_logger()
    assert len(logger.handlers) >= 2
