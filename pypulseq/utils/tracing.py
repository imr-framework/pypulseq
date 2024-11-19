import sys
import traceback

# Global variables indicating whether tracing is enabled and how deep the
# calls are traced.
_tracing: bool = False
_trace_limit: int = 1


def trace_enabled() -> bool:
    """
    Returns whether tracing where sequence events and blocks were created is
    enabled.
    """
    global _tracing
    return _tracing


def enable_trace(limit: int = 1) -> None:
    """
    Enable tracing where sequence events and blocks were created. Note that
    this can slow down sequence creation.

    Parameters
    ----------
    limit : int, optional
        Limit on the stack depth that is stored. Increase when your sequence
        is created in nested functions and you want to trace back to the
        original call.
        The default is 1.
    """
    global _tracing, _trace_limit
    _tracing = True
    _trace_limit = limit


def disable_trace() -> None:
    """
    Disabling tracing where sequence events and blocks were created.
    """
    global _tracing
    _tracing = False


def trace() -> traceback.StackSummary:
    """
    Internal function to fetch a summary of the call stack.

    Returns
    -------
    traceback.StackSummary
        Call stack summary.
    """
    f = sys._getframe().f_back.f_back  # type: ignore
    return traceback.extract_stack(f, limit=_trace_limit)


def format_trace(trace: traceback.StackSummary, indent: int = 0) -> str:
    """
    Format a stack summary into a string. Optionally adds `indent` spaces
    in front of every line.

    Parameters
    ----------
    trace : traceback.StackSummary
        Call stack summary.
    indent : int, optional
        Number of spaces to indent. The default is 0.

    Returns
    -------
    str
        Stack summary formatted into a printable string.
    """
    return '\n'.join(' ' * indent + y for x in trace.format() for y in x.splitlines())
