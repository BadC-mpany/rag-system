import os
import json
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class WinCondition(BaseModel):
    type: str
    targets: Optional[str] = None  # Reference to variable name
    target: Optional[str] = None   # Direct target (for backward compatibility)
    message: str

class InitialState(BaseModel):
    files: List[str]

class FileData(BaseModel):
    name: str
    content: str
    editable: bool = True

class Filesystem(BaseModel):
    files: List[FileData] = []

class Scenario(BaseModel):
    id: str
    name: str
    description: str
    variables: Optional[Dict[str, Any]] = {}
    system_prompt: str
    initial_state: InitialState
    available_tools: List[str]
    user_role: str
    filesystem: Optional[Filesystem] = None
    win_conditions: List[WinCondition]

class ScenarioLoader:
    def __init__(self, scenarios_dir: str = None):
        # Default to the top-level `scenarios` directory in the repo
        if scenarios_dir:
            self.scenarios_dir = scenarios_dir
        else:
            repo_root = os.path.dirname(os.path.dirname(__file__))
            self.scenarios_dir = os.path.join(repo_root, 'scenarios')

    def _process_template_variables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process template variables in the scenario data."""
        variables = data.get('variables', {})
        
        # Create template context
        template_context = variables.copy()
        
        # Special formatting for lists (e.g., forbidden_competitors_list)
        for key, value in variables.items():
            if isinstance(value, list):
                # Create a formatted list string
                if len(value) == 1:
                    template_context[f"{key}_list"] = f'"{value[0]}"'
                elif len(value) == 2:
                    template_context[f"{key}_list"] = f'"{value[0]}" and "{value[1]}"'
                else:
                    formatted_items = [f'"{item}"' for item in value[:-1]]
                    template_context[f"{key}_list"] = f'{", ".join(formatted_items)}, and "{value[-1]}"'
        
        # Process system_prompt template
        if 'system_prompt' in data and isinstance(data['system_prompt'], str):
            try:
                data['system_prompt'] = data['system_prompt'].format(**template_context)
            except KeyError as e:
                print(f"Warning: Template variable {e} not found in variables")
        
        
        return data

    def load_scenario(self, scenario_id: str) -> Scenario:
        """Loads and validates a scenario from a JSON file."""
        filepath = os.path.join(self.scenarios_dir, f"{scenario_id}.json")
        if not os.path.exists(filepath):
            # Fallback to default.json if it exists
            default_path = os.path.join(self.scenarios_dir, 'default.json')
            if os.path.exists(default_path):
                print(f"Scenario '{scenario_id}' not found. Falling back to 'default.json'.")
                filepath = default_path
            else:
                raise ValueError(f"Scenario '{scenario_id}' not found and no default.json present.")

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Process template variables
            data = self._process_template_variables(data)
            
            return Scenario(**data)
        except Exception as e:
            raise ValueError(f"Error loading or validating scenario '{scenario_id}': {e}")

    def list_scenarios(self) -> List[Dict[str, Any]]:
        """Lists all available scenarios."""
        scenarios = []
        for filename in os.listdir(self.scenarios_dir):
            if filename.endswith(".json"):
                try:
                    scenario = self.load_scenario(filename.replace(".json", ""))
                    scenarios.append({
                        "id": scenario.id,
                        "name": scenario.name,
                        "description": scenario.description
                    })
                except ValueError as e:
                    print(f"Warning: Could not load scenario '{filename}': {e}")
        return scenarios
