import json
import os
from typing import List, Optional

from config.settings import (ADMIN_DATA_DIR, CREDENTIALS_FILE,
                             PUBLIC_DATA_DIR, WORKER_DATA_DIR)
from users.schema import User
from utils.security import verify_password


def load_users() -> List[User]:
    """Loads users from the credentials file."""
    if not os.path.exists(CREDENTIALS_FILE):
        return []
    with open(CREDENTIALS_FILE, 'r') as f:
        user_data = json.load(f)
    return [User(**data) for data in user_data]

USERS = load_users()

def authenticate_user(username: str, code: str) -> Optional[User]:
    """
    Authenticates a user by username and code.
    Returns the User object if authentication is successful, otherwise None.
    """
    for user in USERS:
        if user.username == username and verify_password(code, user.code):
            return user
    return None

def authorize_access(user: User, filepath: str) -> bool:
    """
    Authorizes a user's access to a file based on their role.
    """
    filepath = os.path.abspath(filepath)
    
    if user.role == 'admin':
        return True
    
    if user.role == 'manager':
        return filepath.startswith(os.path.abspath(PUBLIC_DATA_DIR)) or \
               filepath.startswith(os.path.abspath(WORKER_DATA_DIR))
               
    if user.role == 'worker':
        return filepath.startswith(os.path.abspath(PUBLIC_DATA_DIR)) or \
               filepath.startswith(os.path.abspath(WORKER_DATA_DIR))
        
    if user.role == 'public':
        return filepath.startswith(os.path.abspath(PUBLIC_DATA_DIR))
        
    return False

def get_accessible_directories(user: User) -> List[str]:
    """
    Returns a list of directories accessible to the user based on their role.
    """
    if user.role == 'admin':
        return [PUBLIC_DATA_DIR, WORKER_DATA_DIR, ADMIN_DATA_DIR]
    if user.role == 'manager':
        return [PUBLIC_DATA_DIR, WORKER_DATA_DIR, ADMIN_DATA_DIR]
    if user.role == 'worker':
        return [PUBLIC_DATA_DIR, WORKER_DATA_DIR]
    if user.role == 'public':
        return [PUBLIC_DATA_DIR]
    return []
