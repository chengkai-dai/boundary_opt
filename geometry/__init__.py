"""Shared triangle-mesh data and topology."""

from .mesh import Mesh, boundary_loop, load_obj, normalize_mesh

__all__ = ["Mesh", "boundary_loop", "load_obj", "normalize_mesh"]
