import argparse
import sys
import os
import random
from typing import Dict
from pathlib import Path
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar
from surfaces_agent.agent.session import global_state as state

class AlloyGenerationSchema(BaseModel):
    input_ref_id: str = Field(..., description="The state reference ID or file path of the bulk/slab structure.")
    target_species: str = Field(..., description="The element symbol to replace (e.g., 'Ti').")
    substitutions: Dict[str, float] = Field(..., description="Dictionary of new elements and their target atomic fractions (e.g., {'Zr': 0.5, 'Ce': 0.5}). Must sum to 1.0.")

def create_random_alloy(input_ref_id: str, target_species: str, substitutions: Dict[str, float]) -> str:
    """
    Solid Solution / High-Entropy Tool: Replaces a specific element in the structure with a randomized mixture of elements.
    
    This tool is used to create High-Entropy Oxides (HEOs), doped materials, or solid solutions. It:
    1. Loads the structure from a file or agent state.
    2. Identifies all sites containing the target_species.
    3. Calculates the exact number of atoms needed for each new element based on the provided fractions.
    4. Randomly shuffles the new elements across the target sites.
    5. Saves the alloyed structure to the agent's state and exports a VASP POSCAR file.
    
    Use this when the user asks to 'dope', 'create a solid solution', or 'make a high entropy alloy'.
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

    target_indices = [i for i, site in enumerate(struct) if site.specie.symbol == target_species]
    if not target_indices:
        return f"Error: No {target_species} atoms found in the structure."

    total_sites = len(target_indices)
    
    # Verify fractions sum to ~1.0
    if abs(sum(substitutions.values()) - 1.0) > 1e-4:
        return f"Error: Substitution fractions must sum to 1.0. Got {sum(substitutions.values())}."

    # Calculate integer atom counts
    counts = {}
    remaining_sites = total_sites
    
    # Sort by fraction descending to minimize rounding errors on major components
    sorted_subs = sorted(substitutions.items(), key=lambda x: x[1], reverse=True)
    
    for i, (el, frac) in enumerate(sorted_subs):
        if i == len(sorted_subs) - 1:
            counts[el] = remaining_sites # Give the rest to the last element
        else:
            count = int(round(frac * total_sites))
            counts[el] = count
            remaining_sites -= count

    if remaining_sites < 0:
        return f"Error: Cannot cleanly partition {total_sites} sites with the given fractions."

    # Create shuffled list of new species
    new_species_list = []
    for el, count in counts.items():
        new_species_list.extend([el] * count)
        
    random.shuffle(new_species_list)

    # Apply substitutions
    for idx, new_specie in zip(target_indices, new_species_list):
        struct.replace(idx, new_specie)

    struct.sort() # Group by element for cleaner POSCAR

    formula = struct.composition.reduced_formula
    ref_id = state.save(struct, prefix=f"alloy_{formula}")
    
    out_dir = Path("workspace")
    out_dir.mkdir(exist_ok=True)
    filename = out_dir / f"{formula}_alloy.vasp"
    
    Poscar(struct).write_file(str(filename))

    report = f"✅ Random Alloy Created.\n"
    report += f"- Replaced {total_sites} '{target_species}' sites with:\n"
    for el, count in counts.items():
        report += f"  - {el}: {count} atoms\n"
    report += f"- New Formula: {formula}\n"
    report += f"- Saved State ID: '{ref_id}'\n"
    report += f"- Exported to: {filename}"

    return report

def main():
    parser = argparse.ArgumentParser(description="Generate a random alloy / solid solution.")
    parser.add_argument("--input", type=str, required=True, help="Input structure file (POSCAR/CIF).")
    parser.add_argument("--target", type=str, required=True, help="Element to replace (e.g., Ti).")
    parser.add_argument("--substitutions", type=str, required=True, help="Comma-separated element:fraction pairs (e.g., Zr:0.5,Ce:0.5).")
    args = parser.parse_args()
    
    subs = {}
    try:
        for pair in args.substitutions.split(","):
            el, frac = pair.split(":")
            subs[el.strip()] = float(frac.strip())
    except ValueError:
        print("Error: Substitutions must be in format El:frac,El:frac (e.g., Zr:0.5,Ce:0.5)")
        sys.exit(1)
        
    print(create_random_alloy(args.input, args.target, subs))

if __name__ == "__main__":
    main()