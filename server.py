from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from platform_logic.scenario_loader import ScenarioLoader
from platform_logic.state_manager import StateManager

app = FastAPI()
scenario_loader = ScenarioLoader()
state_manager = StateManager()

class StartSessionRequest(BaseModel):
    scenario_id: str

@app.get("/")
def read_root():
    return {"message": "Security Testing Platform API is running."}

@app.get("/scenarios")
def list_scenarios():
    return scenario_loader.list_scenarios()

@app.post("/session/start")
def start_session(request: StartSessionRequest):
    try:
        scenario = scenario_loader.load_scenario(request.scenario_id)
        session_state = state_manager.new_session(scenario)
        return {"session_id": session_state.session_id}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
