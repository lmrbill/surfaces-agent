# surfaces_agent/agent/engine.py
import argparse
import sys
import os
import json
from datetime import datetime
from pathlib import Path
from google import genai
from google.genai import types
from dotenv import load_dotenv

# Tool imports
from surfaces_agent.tools.mp import fetch_materials_project_structure
from surfaces_agent.tools.slab import generate_surface_slab
from surfaces_agent.tools.adsorption import enumerate_adsorption_sites
from surfaces_agent.tools.io import save_structure
from surfaces_agent.tools.search import search_scientific_knowledge
from surfaces_agent.tools.analysis import analyze_electronic_properties
from surfaces_agent.tools.md import run_md_simulation
from surfaces_agent.tools.supercell import expand_structure_to_supercell
from surfaces_agent.tools.vacancy import create_surface_vacancy
from surfaces_agent.tools.neb import prepare_neb_pathway
from surfaces_agent.tools.alloy import create_random_alloy
from surfaces_agent.tools.interstitial import create_random_interstitials
from surfaces_agent.agent.session import global_state as session

def main():
    # Load .env first to ensure all API keys are available
    load_dotenv()
    
    parser = argparse.ArgumentParser(description="Run the Surfaces Agent.")
    parser.add_argument("--model", type=str, help="Override AGENT_MODEL from .env")
    args = parser.parse_args()

    api_key = os.environ.get("API_KEY")
    model_id = args.model or os.environ.get("AGENT_MODEL", "gemini-3.1-flash-lite-preview")

    if not api_key:
        print("Error: 'API_KEY' environment variable is not set. Please check your .env file.")
        sys.exit(1)

    # Setup Logging
    workspace_dir = Path("workspace")
    log_dir = workspace_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    
    date_str = datetime.now().strftime("%Y_%m_%d")
    session_log = log_dir / f"session_{date_str}.json"
    
    def log_interaction(role, content):
        entry = {
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content
        }
        with open(session_log, "a") as f:
            f.write(json.dumps(entry) + "\n")

    client = genai.Client(api_key=api_key)
    print(f"🤖 surfaces-agent initialized with {model_id}. Type 'exit' to quit.")
    print(f"Session log: {session_log}")

    agent_tools = [
        fetch_materials_project_structure,
        generate_surface_slab,
        enumerate_adsorption_sites,
        save_structure,
        search_scientific_knowledge,
        analyze_electronic_properties,
        run_md_simulation,
        expand_structure_to_supercell,
        create_surface_vacancy,
        prepare_neb_pathway,
        create_random_alloy,
        create_random_interstitials
    ]

    system_instruction = """You are a computational surface scientist specializing in oxide surfaces and catalytic reactions.
You help researchers perform simulations by reasoning about surface chemistry and calling scientific tools.

CRITICAL RULES:
1. **Always explain your reasoning.** Do not act like a black-box script. Propose a plan and ask for confirmation before executing long chains of tools (e.g., "Plan:\n1. Fetch structure\n2. Cleave slab\nProceed?").
2. **Never fabricate simulation results.** Simulation results must come ONLY from tool outputs. If a result is not from a tool, refuse to answer or explicitly state it is a theoretical estimation.
3. **Workspace Context.** You operate within the `workspace/` directory. If the user provides a file path (e.g., 'workspace/SrTiO3.cif'), use that file directly. Do not re-download from Materials Project if a local file is specified.
4. **Summarize Results.** Raw tool outputs are for you to read. When responding to the user, synthesize and summarize the key scientific metrics (e.g., Energy, Surface Terminations, Max force, p-band center) in a readable format.
5. **State Management.** You have access to persistent session state. When a tool returns a State ID (e.g., 'slab_...'), you can pass that ID directly to subsequent tools without writing to disk intermediate files unless requested.
"""

    chat = client.chats.create(
        model=model_id,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            tools=agent_tools,
            temperature=0.1,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
        )
    )

    while True:
        try:
            prompt = input("\n>> ")
            if not prompt:
                continue
            if prompt.strip().lower() in ['exit', 'quit']:
                break
            
            log_interaction("user", prompt)
            print("🤖 Processing...")
            response = chat.send_message(prompt)
            
            text_response = response.text or "[Agent completed task with tool calls]"
            print(f"\n{text_response}")
            log_interaction("agent", text_response)
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            error_msg = f"Agent engine failed: {e}"
            print(error_msg)
            log_interaction("system_error", error_msg)

if __name__ == "__main__":
    main()