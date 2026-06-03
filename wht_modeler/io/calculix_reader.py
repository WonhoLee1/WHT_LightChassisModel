import re
from typing import List, Dict, Optional, Tuple

from .base_reader import BaseFEMReader
from ..wht_mesh_model import WHTMeshModel
from ..wht_entities import WHTLoadEntry
from wht_solver.load_cases import WHTLoadCase, WHTBCEntry, WHTForceEntry


class CalculixReader(BaseFEMReader):
    """
    Parses a CalculiX .inp file and returns a populated WHTMeshModel.
    Supports basic nodes, elements, sets, materials, and multiple loadcases (STEPs).
    """

    def __init__(self):
        self.model: Optional[WHTMeshModel] = None
        
        # Temporary parsing state
        self._current_step: Optional[WHTLoadCase] = None
        self._current_material_id = 0
        self._material_name_to_id: Dict[str, int] = {}
        self._current_property_id = 0
        
        # Set parsing
        self._current_set_name: Optional[str] = None
        self._current_set_type: Optional[str] = None # 'NSET' or 'ELSET'
        self._current_set_ids: List[int] = []

    def read(self, file_path: str) -> WHTMeshModel:
        self.model = WHTMeshModel(name="CalculiX_Model")
        
        # Encoding fallback support (utf-8 -> utf-16 -> cp949)
        lines = []
        for enc in ["utf-8", "utf-16", "cp949"]:
            try:
                with open(file_path, "r", encoding=enc) as f:
                    lines = f.readlines()
                break
            except UnicodeDecodeError:
                continue
        if not lines:
            raise ValueError(f"Failed to read {file_path} with supported encodings (utf-8, utf-16, cp949)")

        current_keyword = None
        keyword_args = {}

        i = 0
        n_lines = len(lines)
        
        while i < n_lines:
            line = lines[i].strip()
            
            # Skip empty lines or comments
            if not line or line.startswith("**"):
                i += 1
                continue
                
            # Check for keyword
            if line.startswith("*"):
                # End previous set if we were parsing one
                if self._current_set_name is not None:
                    self._save_current_set()
                
                parts = line[1:].split(",")
                current_keyword = parts[0].strip().upper()
                
                keyword_args = {}
                for part in parts[1:]:
                    if "=" in part:
                        k, v = part.split("=", 1)
                        keyword_args[k.strip().upper()] = v.strip()
                
                # Handle single-line keyword logic
                if current_keyword == "STEP":
                    step_name = keyword_args.get("NAME", f"Step_{len(self.model.load_cases) + 1}")
                    self._current_step = WHTLoadCase(name=step_name)
                    
                elif current_keyword == "END STEP":
                    if self._current_step is not None:
                        self.model.load_cases.append(self._current_step)
                        self._current_step = None
                        
                elif current_keyword == "NSET":
                    self._current_set_name = keyword_args.get("NSET", "UNKNOWN_NSET")
                    self._current_set_type = "NSET"
                    self._current_set_ids = []
                    
                elif current_keyword == "ELSET":
                    self._current_set_name = keyword_args.get("ELSET", "UNKNOWN_ELSET")
                    self._current_set_type = "ELSET"
                    self._current_set_ids = []
                    
                elif current_keyword == "MATERIAL":
                    mat_name = keyword_args.get("NAME", "DEFAULT_MAT")
                    self._current_material_id += 1
                    self._material_name_to_id[mat_name] = self._current_material_id
                    # Default material props
                    self.model.add_material(self._current_material_id, E=210000.0, nu=0.3, rho=7.8e-9)
                    
                elif current_keyword == "SHELL SECTION":
                    elset_name = keyword_args.get("ELSET", "")
                    mat_name = keyword_args.get("MATERIAL", "")
                    mat_id = self._material_name_to_id.get(mat_name, 1)
                    # We need the thickness from the next line
                    i += 1
                    if i < n_lines and not lines[i].strip().startswith("*"):
                        t_val = lines[i].strip().split(",")[0].strip()
                        thickness = float(t_val) if t_val else 1.0
                        self._current_property_id += 1
                        self.model.add_property(self._current_property_id, "SHELL", thickness, mat_id)
                        
                        # Assign this property to elements in the elset
                        # (Normally we should wait until all elements are read, but assume ordered for now or we map it)
                        # We will skip strict property assignment to elements for now as wht_mesh_model assigns pid=0 by default
                
                i += 1
                continue
            
            # Parse data lines depending on current_keyword
            if current_keyword == "NODE":
                parts = line.split(",")
                if len(parts) >= 4:
                    nid = int(parts[0])
                    x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                    self.model.add_node(nid, x, y, z)
                    
            elif current_keyword == "ELEMENT":
                parts = line.split(",")
                if len(parts) >= 4:
                    eid = int(parts[0])
                    nids = [int(p) for p in parts[1:]]
                    etype_ccx = keyword_args.get("TYPE", "").upper()
                    # Map CalculiX types to our types
                    etype = "QUAD4"
                    if "S4" in etype_ccx or "C3D8" in etype_ccx:
                        etype = "QUAD4" if len(nids) == 4 else "HEXA8"
                    elif "S3" in etype_ccx or "C3D4" in etype_ccx:
                        etype = "TRIA3" if len(nids) == 3 else "TETRA4"
                    
                    self.model.add_element(eid, nids, etype)
                    
            elif current_keyword in ["NSET", "ELSET"]:
                parts = line.split(",")
                for p in parts:
                    p = p.strip()
                    if p:
                        # Handle GENERATE
                        if "GENERATE" in keyword_args:
                            # Not fully implemented generate, usually requires start, end, step
                            pass
                        else:
                            self._current_set_ids.append(int(p))
                            
            elif current_keyword == "BOUNDARY":
                parts = line.split(",")
                if len(parts) >= 2:
                    target = parts[0].strip()
                    first_dof = int(parts[1].strip())
                    last_dof = int(parts[2].strip()) if len(parts) > 2 and parts[2].strip() else first_dof
                    value = float(parts[3].strip()) if len(parts) > 3 and parts[3].strip() else 0.0
                    
                    dofs = tuple(range(first_dof - 1, last_dof)) # 0-indexed internally
                    
                    nids = self._resolve_target(target)
                    
                    if self._current_step is not None:
                        # Step-specific BC
                        for nid in nids:
                            self._current_step.bcs.append(WHTBCEntry(nid, dofs, value))
                    else:
                        # Global BC
                        self.model.apply_spc(nids, dofs, value)
                        
            elif current_keyword == "CLOAD":
                parts = line.split(",")
                if len(parts) >= 3:
                    target = parts[0].strip()
                    dof = int(parts[1].strip()) - 1 # 0-indexed
                    value = float(parts[2].strip())
                    
                    nids = self._resolve_target(target)
                    
                    if self._current_step is not None:
                        for nid in nids:
                            vec = [0.0] * 6
                            if 0 <= dof < 6:
                                vec[dof] = value
                            self._current_step.forces.append(WHTForceEntry(nid, tuple(vec)))
                    else:
                        for nid in nids:
                            vec = [0.0] * 6
                            if 0 <= dof < 6:
                                vec[dof] = value
                            self.model.loads.append(WHTLoadEntry(nid, tuple(vec)))
            
            # Other data lines (like ELASTIC, DENSITY) can be parsed if needed, but we use defaults for now
            
            i += 1
            
        # If file ends while parsing a set
        if self._current_set_name is not None:
            self._save_current_set()

        return self.model

    def _save_current_set(self):
        if self._current_set_type == "NSET" and self._current_set_name:
            self.model.add_node_set_by_name(self._current_set_name, self._current_set_ids)
        elif self._current_set_type == "ELSET" and self._current_set_name:
            self.model.add_elem_set_by_name(self._current_set_name, self._current_set_ids)
            
        self._current_set_name = None
        self._current_set_type = None
        self._current_set_ids = []

    def _resolve_target(self, target_str: str) -> List[int]:
        """Resolve a node ID or a node set name to a list of node IDs."""
        if target_str.isdigit():
            return [int(target_str)]
        else:
            try:
                return self.model.get_nodes_by_set_name(target_str)
            except KeyError:
                return []
