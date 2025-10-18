import os
import yaml
from pydantic import BaseModel
from typing import List, Dict, Any

class WinCondition(BaseModel):
    type: str
    target: str
    message: str

class InitialState(BaseModel):
    files: List[str]

class Scenario(BaseModel):
    id: str
    name: str
    description: str
    initial_state: InitialState
    agent_prompt: str
    available_tools: List[str]
    user_role: str
    win_conditions: List[WinCondition]

class ScenarioLoader:
    def __init__(self, scenarios_dir: str = None):
        # Default to the top-level `scenarios` directory in the repo
        if scenarios_dir:
            self.scenarios_dir = scenarios_dir
        else:
            repo_root = os.path.dirname(os.path.dirname(__file__))
            self.scenarios_dir = os.path.join(repo_root, 'scenarios')

    def load_scenario(self, scenario_id: str) -> Scenario:
        """Loads and validates a scenario from a YAML file."""
        filepath = os.path.join(self.scenarios_dir, f"{scenario_id}.yaml")
        if not os.path.exists(filepath):
            # Fallback to default.yaml if it exists
            default_path = os.path.join(self.scenarios_dir, 'default.yaml')
            if os.path.exists(default_path):
                print(f"Scenario '{scenario_id}' not found. Falling back to 'default.yaml'.")
                filepath = default_path
            else:
                raise ValueError(f"Scenario '{scenario_id}' not found and no default.yaml present.")

        try:
            with open(filepath, 'r') as f:
                data = yaml.safe_load(f)
            return Scenario(**data)
        except Exception as e:
            raise ValueError(f"Error loading or validating scenario '{scenario_id}': {e}")

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """Lists all available scenarios."""
        scenarios = []
        for filename in os.listdir(self.scenarios_dir):
            if filename.endswith(".yaml"):
                try:
                    scenario = self.load_scenario(filename.replace(".yaml", ""))
                    scenarios.append({
                        "id": scenario.id,
                        "name": scenario.name,
                        "description": scenario.description
                    })
                except ValueError as e:
                    print(f"Warning: Could not load scenario '{filename}': {e}")
        return scenarios
import os
