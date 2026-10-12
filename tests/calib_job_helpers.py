"""Module-level worker jobs for `solve_many` tests (spawn pickles functions by reference)."""
import os


def die_on_width_666(args):
    """Stand-in for `_solve_job`: the worker process dies hard (OOM kill, segfault) on one input."""
    _code_dir, _kp, _ln, w, _h, _kw = args
    if w == 666:
        os._exit(1)
    return None, None
