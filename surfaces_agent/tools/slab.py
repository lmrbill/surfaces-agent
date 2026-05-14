import argparse
import sys
import os
import contextlib
import warnings
from typing import List
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core.surface import SlabGenerator
from pymatgen.core import Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from surfaces_agent.agent.session import global_state as state

@contextlib.contextmanager
def suppress_output():
    """Context manager to silence CHGNet/PyTorch/C-level initialization logs."""
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                yield

class SlabRelaxationSchema(BaseModel):
    bulk_ref_id: str = Field(..., description="The state reference ID or file path of the bulk structure.")
    miller: List[int] = Field(..., description="Miller indices for the surface cleave (e.g., [0, 0, 1]).")
    min_slab_size: float = Field(15.0, description="Minimum slab thickness in Angstroms.")
    min_vacuum: float = Field(15.0, description="Minimum vacuum thickness in Angstroms.")
    relax: bool = Field(True, description="Whether to relax the slab using CHGNet. Set to False for pure geometric cleaving.")
    all_terminations: bool = Field(False, description="Whether to generate and return all unique terminations.")

def get_surface_termination(slab) -> str:
    """Dynamically identifies the elemental species in the topmost layer of the slab."""
    max_z = max(site.frac_coords[2] for site in slab)
    top_layer = [site for site in slab if np.isclose(site.frac_coords[2], max_z, atol=0.05)]
    species = sorted(list(set(site.specie.symbol for site in top_layer)))
    return "-".join(species) + " terminated"

def generate_surface_slab(
    bulk_ref_id: str, 
    miller: List[int], 
    min_slab_size: float = 10.0, 
    min_vacuum: float = 15.0,
    relax: bool = True,
    all_terminations: bool = False
) -> str:
    """
    Core Surface Creation Tool: Cleaves a bulk crystal along specific Miller indices.
    Can return all unique terminations, and optionally relaxes the resulting slab(s) using CHGNet ML-interatomic potentials.
    """
    
    if os.path.isfile(bulk_ref_id):
        try:
            bulk_struct = Structure.from_file(bulk_ref_id)
            print(f"   [Tool] Loaded bulk directly from file: {bulk_ref_id}")
        except Exception as e:
            return f"Error parsing structure file '{bulk_ref_id}': {str(e)}"
    else:
        try:
            bulk_struct = state.load(bulk_ref_id)
            print(f"   [Tool] Loaded bulk from agent state ID: {bulk_ref_id}")
        except KeyError:
            return f"Error: '{bulk_ref_id}' is not a valid file or state ID."

    formula = bulk_struct.composition.reduced_formula
    miller_str = f"({miller[0]}{miller[1]}{miller[2]})"
    
    e_bulk_per_atom = None
    optimizer = None
    chgnet = None

    if relax:
        try:
            with suppress_output():
                from chgnet.model.model import CHGNet
                from chgnet.model.dynamics import StructOptimizer
        except ImportError:
            return "Error: CHGNet or PyTorch is not installed. Run with relax=False."

        print(f"   [Tool] Initializing CHGNet...")
        with suppress_output():
            chgnet = CHGNet.load()
            optimizer = StructOptimizer(model=chgnet)

        # 1. Relax Bulk (Cell + Atoms)
        print(f"   [Tool] Relaxing bulk {formula} (Cell+Atoms)...")
        with suppress_output():
            bulk_relax = optimizer.relax(bulk_struct, relax_cell=True, verbose=False)
        
        relaxed_bulk = bulk_relax["final_structure"]
        with suppress_output():
            bulk_energy = chgnet.predict_structure(relaxed_bulk)["e"]
        e_bulk_per_atom = bulk_energy / len(relaxed_bulk)
    else:
        relaxed_bulk = bulk_struct

    # 2. Generate and Analyze Slab
    print(f"   [Tool] Cleaving {miller_str} surface...")
    slabgen = SlabGenerator(
        initial_structure=relaxed_bulk,
        miller_index=miller,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum,
        center_slab=True
    )
    
    # Generate unique slabs by scanning shifts
    matcher = StructureMatcher(ltol=0.1, stol=0.1, angle_tol=1)
    unique_slabs = []
    
    for shift in np.linspace(0, 1, 20):
        try:
            s = slabgen.get_slab(shift=shift)
            is_unique = True
            for us in unique_slabs:
                if matcher.fit(s, us):
                    is_unique = False
                    break
            if is_unique:
                unique_slabs.append(s)
        except Exception:
            pass
            
    if not unique_slabs:
        return f"Error: Could not generate any slabs for {miller_str}."

    slabs_to_process = unique_slabs if all_terminations else [unique_slabs[0]]
    
    out_dir = Path("workspace")
    out_dir.mkdir(exist_ok=True)
    
    output_messages = []
    
    for idx, slab in enumerate(slabs_to_process):
        term_type = get_surface_termination(slab)

        # 3. Explicit Kinematics & FixAtoms via ASE
        z_coords = [site.coords[2] for site in slab]
        mid_z = min(z_coords) + (max(z_coords) - min(z_coords)) / 2.0
        
        selective_dynamics = []
        fixed_indices = []
        for i, site in enumerate(slab):
            if site.coords[2] < mid_z:
                selective_dynamics.append([False, False, False]) # Fixed
                fixed_indices.append(i)
            else:
                selective_dynamics.append([True, True, True])    # Free
                
        slab.add_site_property("selective_dynamics", selective_dynamics)

        if relax:
            print(f"   [Tool] Relaxing slab {idx+1}/{len(slabs_to_process)} ({len(slab)} atoms, {term_type})...")
            print(f"   [Tool] Kinematics: Bottom {len(fixed_indices)} atoms fixed (z < {mid_z:.2f} Å).")
            
            with suppress_output():
                from pymatgen.io.ase import AseAtomsAdaptor
                from ase.constraints import FixAtoms
                
                slab_ase = AseAtomsAdaptor.get_atoms(slab)
                slab_ase.set_constraint(FixAtoms(indices=fixed_indices))
                
                slab_relax = optimizer.relax(slab_ase, relax_cell=False, verbose=False)
            
            final_slab = slab_relax["final_structure"]
            final_slab.add_site_property("selective_dynamics", selective_dynamics)
            
            with suppress_output():
                slab_energy = chgnet.predict_structure(final_slab)["e"]

            # 4. Energy Calculation
            area = slab.surface_area
            n_atoms = len(final_slab)
            surface_energy_j_m2 = ((slab_energy - (n_atoms * e_bulk_per_atom)) / (2 * area)) * 16.02176
        else:
            final_slab = slab
            surface_energy_j_m2 = None

        # 5. Save State and Local File
        ref_id = state.save(final_slab, prefix=f"slab_{formula}_{miller[0]}{miller[1]}{miller[2]}_{term_type}")
        
        from pymatgen.io.vasp import Poscar
        filename = out_dir / f"{formula}_{miller[0]}{miller[1]}{miller[2]}_{term_type}_{idx}_"
        filename = filename.with_name(filename.name + ("relaxed.vasp" if relax else "unrelaxed.vasp"))
        Poscar(final_slab).write_file(str(filename))

        msg = (
            f"Slab {idx+1}:\n"
            f"- Termination: {term_type}\n"
        )
        if relax:
            msg += f"- Surface Energy (CHGNet): {surface_energy_j_m2:.3f} J/m²\n"
        msg += (
            f"- State ID: '{ref_id}'\n"
            f"- Saved to: {filename}\n"
        )
        output_messages.append(msg)

    final_output = f"Successfully generated {len(slabs_to_process)} {formula} {miller_str} slab(s).\n\n"
    final_output += "\n".join(output_messages)
    
    if relax:
        final_output += (
            "\nAGENT INSTRUCTION: Please use your Google Search tool to find experimental or DFT literature values "
            f"(with DOI) for the surface energy of {formula} {miller_str} and compare it to the calculated value above."
        )
        
    return final_output

def main():
    parser = argparse.ArgumentParser(description="Generate and relax a surface slab from a bulk structure.")
    parser.add_argument("--bulk-file", type=str, required=True, help="Path to the bulk structure file (e.g., CIF, POSCAR).")
    parser.add_argument("--miller", type=int, nargs=3, required=True, help="Miller indices (e.g., 0 0 1).")
    parser.add_argument("--min-slab-size", type=float, default=10.0, help="Minimum slab thickness in Angstroms.")
    parser.add_argument("--min-vacuum", type=float, default=15.0, help="Minimum vacuum size in Angstroms.")
    parser.add_argument("--no-relax", action="store_true", help="Skip CHGNet relaxation.")
    parser.add_argument("--all-terminations", action="store_true", help="Generate all unique terminations.")
    
    args = parser.parse_args()
    
    try:
        result = generate_surface_slab(
            args.bulk_file, 
            args.miller, 
            args.min_slab_size, 
            args.min_vacuum,
            relax=not args.no_relax,
            all_terminations=args.all_terminations
        )
        print(result)
    except Exception as e:
        print(f"Fatal Tool Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()