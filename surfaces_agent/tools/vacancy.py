# surfaces_agent/tools/vacancy.py
import argparse
import sys
import os
from pathlib import Path
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from typing import Optional
from surfaces_agent.agent.session import global_state as state

class VacancyGenerationSchema(BaseModel):
    input_ref_id: str = Field(..., description="The state reference ID or file path of the slab/bulk.")
    species: str = Field("O", description="The element symbol to remove (e.g., 'O').")
    site_index: Optional[int] = Field(None, description="1-based index of the atom to remove. If None, uses num_vacancies.")
    num_vacancies: int = Field(1, description="Number of random vacancies to create if site_index is not provided. If 1, removes the topmost atom by default, unless random is specified.")
    random_selection: bool = Field(False, description="If True, removes atoms randomly instead of the topmost atom.")

def create_surface_vacancy(input_ref_id: str, species: str = "O", site_index: Optional[int] = None, num_vacancies: int = 1, random_selection: bool = False) -> str:
    """
    Defect Engineering Tool: Creates single or multiple atom vacancies (e.g., Oxygen vacancy) on a surface or bulk.
    
    This tool is used to simulate Mars-van Krevelen mechanisms, defect site reactivity, or non-stoichiometry. It:
    1. Loads the structure from a file or agent state.
    2. Identifies the target atoms to remove (either by explicit index, topmost atom, or random selection).
    3. Removes the atoms and preserves the Selective Dynamics constraints of the remaining atoms.
    4. Saves the defective structure to the agent's state and exports a VASP POSCAR file.
    
    Use this when the user asks to 'create an oxygen vacancy', 'remove the surface O', 'calculate Evac', or 'create non-stoichiometric bulk'.
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

    target_indices = [i for i, site in enumerate(struct) if site.specie.symbol == species]
    if not target_indices:
        return f"Error: No {species} atoms found in the structure."
    
    if num_vacancies > len(target_indices):
        return f"Error: Requested {num_vacancies} vacancies but only {len(target_indices)} {species} atoms exist."

    indices_to_remove = []

    if site_index is not None:
        idx_to_remove = site_index - 1
        if idx_to_remove not in target_indices:
            return f"Error: Atom at index {site_index} is not {species} or out of bounds."
        indices_to_remove.append(idx_to_remove)
    elif random_selection or num_vacancies > 1:
        import random
        indices_to_remove = random.sample(target_indices, num_vacancies)
    else:
        # Default behavior: remove the topmost atom (highest z)
        highest_z = -1e9
        idx_to_remove = -1
        for idx in target_indices:
            if struct[idx].coords[2] > highest_z:
                highest_z = struct[idx].coords[2]
                idx_to_remove = idx
        indices_to_remove.append(idx_to_remove)

    removed_coords = [struct[i].coords for i in indices_to_remove]
    
    # Sort indices in reverse order so deleting doesn't shift earlier indices
    struct.remove_sites(sorted(indices_to_remove, reverse=True))
    
    formula = struct.composition.reduced_formula
    ref_id = state.save(struct, prefix=f"vac_{species}_{formula}")
    
    out_dir = Path("workspace")
    out_dir.mkdir(exist_ok=True)
    filename = out_dir / f"{formula}_vac_{species}.vasp"
    Poscar(struct).write_file(str(filename))

    msg = f"✅ Vacancy Created.\n- Removed {len(indices_to_remove)} {species} atoms:\n"
    for i, c in zip(indices_to_remove, removed_coords):
        msg += f"  - Index {i+1} at {c}\n"
    msg += f"- New Formula: {formula}\n"
    msg += f"- Saved State ID: '{ref_id}'\n"
    msg += f"- Exported to: {filename}"

    return msg

def main():
    parser = argparse.ArgumentParser(description="Generate one or more atom vacancies on a surface slab.")
    parser.add_argument("--input", type=str, required=True, help="Input structure file (POSCAR/CIF).")
    parser.add_argument("--species", type=str, default="O", help="Element to remove (default: O).")
    parser.add_argument("--index", type=int, help="1-based index of atom to remove (default: topmost).")
    parser.add_argument("--num-vacancies", type=int, default=1, help="Number of random vacancies to create.")
    parser.add_argument("--random", action="store_true", help="Remove atoms randomly.")
    args = parser.parse_args()
    
    print(create_surface_vacancy(args.input, args.species, args.index, args.num_vacancies, args.random))

if __name__ == "__main__":
    main()