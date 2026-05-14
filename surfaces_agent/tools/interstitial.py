import argparse
import sys
import os
import random
import numpy as np
from pathlib import Path
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from surfaces_agent.agent.session import global_state as state

class InterstitialGenerationSchema(BaseModel):
    input_ref_id: str = Field(..., description="The state reference ID or file path of the structure.")
    species: str = Field("H", description="The element symbol to add (e.g., 'H').")
    num_atoms: int = Field(1, description="Number of atoms to add.")
    reference_species: str = Field("O", description="The element symbol to attach the interstitial to (e.g., 'O').")
    bond_distance: float = Field(0.98, description="Distance from the reference atom in Angstroms.")

def create_random_interstitials(input_ref_id: str, species: str = "H", num_atoms: int = 1, reference_species: str = "O", bond_distance: float = 0.98) -> str:
    """
    Defect Engineering Tool: Adds random interstitial atoms (like protons) bound to specific reference atoms (like Oxygen).
    
    This tool is used to create hydrated oxides, protonic conductors, or interstitial doped materials. It:
    1. Loads the structure from a file or agent state.
    2. Identifies all sites containing the reference_species (e.g., O).
    3. Randomly selects reference sites and attaches the new species (e.g., H) at a random angle but fixed distance.
    4. Saves the doped structure to the agent's state and exports a VASP POSCAR file.
    
    Use this when the user asks to 'add interstitial protons', 'hydrate the bulk', or 'dope with interstitial'.
    """
    if os.path.isfile(input_ref_id):
        try:
            struct = Structure.from_file(input_ref_id)
        except Exception as e:
            return f"Error parsing file '{input_ref_id}': {str(e)}"
    else:
        try:
            struct = state.load(input_ref_id)
        except KeyError:
            return f"Error: '{input_ref_id}' is not a valid file or state ID."

    target_indices = [i for i, site in enumerate(struct) if site.specie.symbol == reference_species]
    if not target_indices:
        return f"Error: No {reference_species} reference atoms found in the structure to attach to."

    if num_atoms > len(target_indices):
        return f"Error: Requested {num_atoms} interstitials but only {len(target_indices)} {reference_species} reference sites exist."

    chosen_indices = random.sample(target_indices, num_atoms)
    
    added_coords = []
    for ref_idx in chosen_indices:
        ref_site = struct[ref_idx]
        
        # Pick random direction vector in 3D
        phi = random.uniform(0, 2 * np.pi)
        costheta = random.uniform(-1, 1)
        theta = np.arccos(costheta)
        
        direction = np.array([
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(theta)
        ])
        
        new_coord = ref_site.coords + direction * bond_distance
        struct.append(species, new_coord, coords_are_cartesian=True)
        added_coords.append(new_coord)

    struct.sort()

    formula = struct.composition.reduced_formula
    ref_id = state.save(struct, prefix=f"int_{species}_{formula}")
    
    out_dir = Path("workspace")
    out_dir.mkdir(exist_ok=True)
    filename = out_dir / f"{formula}_int_{species}.vasp"
    Poscar(struct).write_file(str(filename))

    msg = f"✅ Interstitials Added.\n- Added {num_atoms} {species} atoms attached to {reference_species}:\n"
    for i, c in enumerate(added_coords):
        msg += f"  - Atom {i+1} at {c}\n"
    msg += f"- New Formula: {formula}\n"
    msg += f"- Saved State ID: '{ref_id}'\n"
    msg += f"- Exported to: {filename}"

    return msg

def main():
    parser = argparse.ArgumentParser(description="Generate random interstitial defects.")
    parser.add_argument("--input", type=str, required=True, help="Input structure file (POSCAR/CIF).")
    parser.add_argument("--species", type=str, default="H", help="Element to add (default: H).")
    parser.add_argument("--num-atoms", type=int, default=1, help="Number of atoms to add.")
    parser.add_argument("--ref-species", type=str, default="O", help="Reference atom to attach to.")
    parser.add_argument("--distance", type=float, default=0.98, help="Bond distance (Angstroms).")
    args = parser.parse_args()
    
    print(create_random_interstitials(args.input, args.species, args.num_atoms, args.ref_species, args.distance))

if __name__ == "__main__":
    main()