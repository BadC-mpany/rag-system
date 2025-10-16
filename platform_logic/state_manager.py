import redis
import pickle
from pydantic import BaseModel
from typing import List, Dict, Any
import uuid

from agent.security_agent import SecurityAgent
from platform_logic.scenario_loader import Scenario

class SessionState(BaseModel):
    session_id: str
    agent: SecurityAgent
    virtual_file_system: Dict[str, str]
    
    class Config:
        arbitrary_types_allowed = True

class StateManager:
    def __init__(self, redis_host: str = 'localhost', redis_port: int = 6379):
        self.redis = redis.Redis(host=redis_host, port=redis_port, db=0)

    def new_session(self, scenario: Scenario) -> SessionState:
        session_id = str(uuid.uuid4())
        agent = SecurityAgent(scenario)
        
        # Initialize virtual file system
        virtual_fs = {}
        for filepath in scenario.initial_state.files:
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    virtual_fs[filepath] = f.read()
            except FileNotFoundError:
                print(f"Warning: Initial state file not found: {filepath}")
        
        state = SessionState(
            session_id=session_id,
            agent=agent,
            virtual_file_system=virtual_fs
        )
        self.save_session(state)
        return state

    def save_session(self, state: SessionState):
        """Serializes and saves the session state to Redis."""
        serialized_state = pickle.dumps(state)
        self.redis.set(state.session_id, serialized_state)

    def load_session(self, session_id: str) -> SessionState:
        """Loads and deserializes a session state from Redis."""
        serialized_state = self.redis.get(session_id)
        if not serialized_state:
            raise ValueError("Session not found.")
        return pickle.loads(serialized_state)
