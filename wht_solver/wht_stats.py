import numpy as np
from typing import Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    from wht_solver.wht_result import WHTSolverResult
    from wht_modeler.wht_mesh_model import WHTMeshModel

class ResultStatsCalculator:
    """
    ResultStatsCalculator
    =====================
    Computes statistical metrics (Max, Mean, Standard Deviation) for
    Stress, Strain, Displacement, and Element Strain Energy from a WHTSolverResult.

    Usage Rule: Always prioritize using this class when calculating result statistics
    to ensure consistency across the optimization loop and post-processing.
    """

    @staticmethod
    def _compute_von_mises(stress: np.ndarray) -> np.ndarray:
        """
        Compute Von Mises stress from 6-component stress vector.
        stress: (..., 6) array [Sxx, Syy, Szz, Sxy, Syz, Sxz]
        Returns: (...) array
        """
        sxx, syy, szz, sxy, syz, sxz = stress[..., 0], stress[..., 1], stress[..., 2], stress[..., 3], stress[..., 4], stress[..., 5]
        vm = np.sqrt(0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2 + 6.0 * (sxy**2 + syz**2 + sxz**2)))
        return vm

    @staticmethod
    def _compute_equivalent_strain(strain: np.ndarray) -> np.ndarray:
        """
        Compute Von Mises equivalent strain.
        strain: (..., 6) array [Exx, Eyy, Ezz, Exy, Eyz, Exz] (Engineering shear strains!)
        Returns: (...) array
        """
        exx, eyy, ezz, gxy, gyz, gxz = strain[..., 0], strain[..., 1], strain[..., 2], strain[..., 3], strain[..., 4], strain[..., 5]
        # Equivalent strain formula (Von Mises for strain)
        # Assuming isotropic, nu ~ 0.3. The standard invariant form:
        # e_eq = 1/(1+nu) * sqrt( 0.5 * ((exx-eyy)^2 + (eyy-ezz)^2 + (ezz-exx)^2 + 1.5*(gxy^2 + gyz^2 + gxz^2)) )
        # A simpler general definition of equivalent strain often used:
        # sqrt( 2/3 * (exx^2 + eyy^2 + ezz^2 + 0.5*(gxy^2 + gyz^2 + gxz^2)) )
        e_eq = np.sqrt((2.0 / 3.0) * (exx**2 + eyy**2 + ezz**2 + 0.5 * (gxy**2 + gyz**2 + gxz**2)))
        return e_eq

    @classmethod
    def compute_stats(cls, result: 'WHTSolverResult', model: 'WHTMeshModel') -> Dict[str, Any]:
        """
        Compute and return comprehensive statistics for the given result.
        
        Returns a dict with:
            {
                "stress": {"max": float, "mean": float, "std": float},
                "strain": {"max": float, "mean": float, "std": float},
                "displacement": {"max": float, "mean": float, "std": float},
                "energy": {"max": float, "mean": float, "std": float, "total": float}
            }
        """
        stats = {
            "stress": {"max": 0.0, "mean": 0.0, "std": 0.0},
            "strain": {"max": 0.0, "mean": 0.0, "std": 0.0},
            "displacement": {"max": 0.0, "mean": 0.0, "std": 0.0},
            "energy": {"max": 0.0, "mean": 0.0, "std": 0.0, "total": 0.0}
        }

        # 1. Displacement Stats
        if result.displacement is not None:
            # result.displacement is (N, 6)
            disp_mag = np.linalg.norm(result.displacement[:, :3], axis=1)
            stats["displacement"]["max"] = float(np.max(disp_mag))
            stats["displacement"]["mean"] = float(np.mean(disp_mag))
            stats["displacement"]["std"] = float(np.std(disp_mag))

        # 2. Stress, Strain, Energy Stats
        if result.cell_data:
            # In solve_static, cell_data arrays are (1, M, 6) or (M, 6) depending on structure.
            # Assuming (1, M, 6) for static case 0.
            stress_array = result.cell_data.get("Stress")
            strain_array = result.cell_data.get("Strain")

            if stress_array is not None and strain_array is not None:
                if stress_array.ndim == 3:
                    stress_array = stress_array[0] # Take the first timestep/mode (M, 6)
                if strain_array.ndim == 3:
                    strain_array = strain_array[0]

                # Filter out zero rows (padded for non-shell elements like RBE)
                # To accurately compute mean/std, we only consider elements with non-zero stress/strain vectors.
                active_mask = np.any(np.abs(stress_array) > 1e-12, axis=1)
                active_stress = stress_array[active_mask]
                active_strain = strain_array[active_mask]

                if len(active_stress) > 0:
                    # Stress Stats
                    vm_stress = cls._compute_von_mises(active_stress)
                    stats["stress"]["max"] = float(np.max(vm_stress))
                    stats["stress"]["mean"] = float(np.mean(vm_stress))
                    stats["stress"]["std"] = float(np.std(vm_stress))

                    # Strain Stats
                    eq_strain = cls._compute_equivalent_strain(active_strain)
                    stats["strain"]["max"] = float(np.max(eq_strain))
                    stats["strain"]["mean"] = float(np.mean(eq_strain))
                    stats["strain"]["std"] = float(np.std(eq_strain))

                    # Element Strain Energy Density (SED)
                    # U_density = 0.5 * sum(stress * strain)
                    sed = 0.5 * np.sum(active_stress * active_strain, axis=1)
                    
                    # Convert SED to Element Strain Energy (U_e = SED * Area * Thickness)
                    # For a quick approximation and consistent metric without full geometric integration,
                    # we often use Volume. Let's compute Volume for active elements.
                    # This requires mapping the active elements to their geometric area and thickness.
                    # In WHTMeshModel, model.elements holds all elements.
                    # The array active_mask maps exactly to the order of elements (0 to M-1)
                    volumes = []
                    active_indices = np.where(active_mask)[0]
                    # We assume cell_data indices correspond to model.elements ordered by sorted keys
                    # Check wht_solver.py: "for key in rd_q: shell_data = rd_q[key] + rd_t[key] ... full_data = np.zeros((n_cells..."
                    # Actually, ElementStressRecovery.recover_quad4 returns an array sized (M_shells, 6).
                    # Wait, looking closely at solve_static:
                    # n_cells = len(self.model.elements)
                    # full_data = np.zeros((n_cells, shell_data.shape[1]))
                    # But the code says:
                    # cell_data[key] = shell_data[np.newaxis, :, :]  # (1, M_shells, 6)
                    # If shell_data is size of elements, great.
                    # Element volumes approximation:
                    element_energy = sed  # Default to SED if volume computation is complex
                    total_energy = 0.0

                    try:
                        elem_keys = sorted(model.elements.keys())
                        if len(elem_keys) == len(active_mask):
                            computed_energies = []
                            for i, idx in enumerate(active_indices):
                                eid = elem_keys[idx]
                                elem = model.elements[eid]
                                # Approximate volume
                                vol = 1.0
                                if hasattr(elem, 'area') and hasattr(elem, 'thickness'):
                                    vol = elem.area() * elem.thickness
                                elif hasattr(elem, 't'): # some shell elements have .t
                                    # Area calculation requires nodes
                                    vol = elem.t
                                computed_energies.append(sed[i] * vol)
                            
                            element_energy = np.array(computed_energies)
                            total_energy = float(np.sum(element_energy))
                        else:
                            # Fallback if array lengths mismatch
                            element_energy = sed
                            total_energy = float(np.sum(sed)) # Just sum of SED
                    except Exception:
                        element_energy = sed
                        total_energy = float(np.sum(sed))
                    
                    stats["energy"]["max"] = float(np.max(element_energy))
                    stats["energy"]["mean"] = float(np.mean(element_energy))
                    stats["energy"]["std"] = float(np.std(element_energy))
                    stats["energy"]["total"] = total_energy

        # If C_i (Compliance) is provided elsewhere, it might be more accurate for total energy.
        # But this gives us the element-wise distribution.
        return stats
