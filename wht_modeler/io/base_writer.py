"""base_writer.py — Abstract base for all FEM writers."""

from abc import ABC, abstractmethod
from ..wht_mesh_model import WHTMeshModel


class BaseFEMWriter(ABC):
    @abstractmethod
    def write(self, model: WHTMeshModel, file_path: str) -> None:
        """Serialize model to file_path."""
