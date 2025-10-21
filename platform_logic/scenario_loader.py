import os
import json
import yaml
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class WinCondition(BaseModel):
    type: str
    targets: Optional[str] = None  # Reference to variable name
    target: Optional[str] = None   # Direct target (for backward compatibility)
    message: str
    score: Optional[str] = None
    caseSensitive: Optional[bool] = True

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
    initial_state: Optional[InitialState] = None
    available_tools: List[str]
    user_role: str
    filesystem: Optional[Filesystem] = None
    win_conditions: List[WinCondition]

class ScenarioLoader:
    def __init__(self, scenarios_dir: str = None):
        if scenarios_dir:
            self.scenarios_dir = scenarios_dir
        else:
            repo_root = os.path.dirname(os.path.dirname(__file__))
            self.scenarios_dir = os.path.join(repo_root, 'scenarios')

    def _process_template_variables(self, data: Dict[str, Any]) -> Dict[str, Any]:
        variables = data.get('variables', {})
        template_context = variables.copy()
        for key, value in variables.items():
            if isinstance(value, list):
                if len(value) == 1:
                    template_context[f"{key}_list"] = f'"{value[0]}"'
                elif len(value) == 2:
                    template_context[f"{key}_list"] = f'"{value[0]}" and "{value[1]}"'
                else:
                    formatted_items = [f'"{item}"' for item in value[:-1]]
                    template_context[f"{key}_list"] = f'{", ".join(formatted_items)}, and "{value[-1]}"'
        
        if 'system_prompt' in data and isinstance(data['system_prompt'], str):
            try:
                data['system_prompt'] = data['system_prompt'].format(**template_context)
            except KeyError as e:
                print(f"Warning: Template variable {e} not found in variables for scenario {data.get('id')}")
        return data

    def load_scenario(self, scenario_id: str) -> Scenario:
        base_path = os.path.join(self.scenarios_dir, scenario_id)
        json_path = f"{base_path}.json"
        yaml_path = f"{base_path}.yaml"

        filepath = None
        if os.path.exists(json_path):
            filepath = json_path
        elif os.path.exists(yaml_path):
            filepath = yaml_path
        else:
            raise ValueError(f"Scenario '{scenario_id}' not found as JSON or YAML.")

        try:
            with open(filepath, 'r') as f:
                if filepath.endswith(".json"):
                    data = json.load(f)
                else:
                    data = yaml.safe_load(f)
            
            data = self._process_template_variables(data)
            return Scenario(**data)
        except Exception as e:
            raise ValueError(f"Error loading or validating scenario '{scenario_id}': {e}")

    def list_scenarios(self) -> List[Dict[str, Any]]:
        scenarios = []
        for filename in os.listdir(self.scenarios_dir):
            if filename.endswith(('.json', '.yaml')):
                scenario_id = filename.rsplit('.', 1)[0]
                try:
                    scenario = self.load_scenario(scenario_id)
                    scenarios.append({
                        "id": scenario.id,
                        "name": scenario.name,
                        "description": scenario.description
                    })
                except ValueError as e:
                    print(f"Warning: Could not load scenario '{filename}': {e}")
        return scenarios
