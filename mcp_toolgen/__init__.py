"""
mcp_toolgen package root.

Exports:
    generate_tools_from_graphql
    generate_tools_from_proto
    __version__
"""

from importlib.metadata import version as _pkg_version
from .mcp_toolgen import (
    generate_tools_from_graphql,
    generate_tools_from_proto,
)

__all__ = [
    "generate_tools_from_graphql",
    "generate_tools_from_proto",
]

__version__ = "0.1.3"
