from backends.base import PuzzleScriptSearchBackend, SearchResult

__all__ = [
    "NodeJSPuzzleScriptBackend",
    "PuzzleScriptSearchBackend",
    "SearchResult",
]


# Lazily import the NodeJS backend so that merely importing `backends` (e.g. via
# `from backends.base import SearchResult`, which the C++ engine does) does NOT
# pull in `backends.nodejs` -> javascript/Node + jax. `from backends import
# NodeJSPuzzleScriptBackend` still works via PEP 562.
def __getattr__(name):
    if name == "NodeJSPuzzleScriptBackend":
        from backends.nodejs import NodeJSPuzzleScriptBackend
        return NodeJSPuzzleScriptBackend
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
