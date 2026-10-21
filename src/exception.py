"""Explicit project error type; normal exceptions keep their Python traceback."""


class TrafficMonitorError(RuntimeError):
    pass
