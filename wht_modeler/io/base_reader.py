"""base_reader.py — Abstract base for all FEM readers."""

from abc import ABC, abstractmethod
from ..wht_mesh_model import WHTMeshModel


class BaseFEMReader(ABC):
    @abstractmethod
    def read(self, file_path: str) -> WHTMeshModel:
        """Parse file_path and return a populated WHTMeshModel."""
