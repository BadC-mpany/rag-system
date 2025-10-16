import logging
import os
from datetime import datetime

from config.settings import LOGS_DIR
from core.access_control import authenticate_user, get_accessible_directories
from agent.security_agent import SecurityAgent
from platform_logic.scenario_loader import ScenarioLoader, Scenario


def setup_logging(username: str):
    """Sets up logging for a user session."""
    log_dir = LOGS_DIR
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f"{username}_history.log")
    
    logging.basicConfig(
        filename=log_file,
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

def main():
    """Main function to run the RAG system."""

    # --- Authentication ---
    username = input("Enter username: ")
    code = input("Enter code: ")
    
    user = authenticate_user(username, code)
    
    if not user:
        print("Authentication failed. Invalid username or code.")
        return

    print(f"Logged in as {user.role}.")
    
    # --- Setup ---
    setup_logging(user.username)
    logging.info(f"User '{user.username}' logged in with role '{user.role}'.")
    
    print("Initializing agent...")
    
    # Create a default scenario for the user
    accessible_dirs = get_accessible_directories(user)
    scenario = Scenario(
        id="default",
        name="Default RAG Assistant",
        description="A general-purpose corporate assistant",
        initial_state={"files": accessible_dirs},
        agent_prompt="You are a helpful corporate assistant for InnovateX.",
        available_tools=["document_search", "url_parser"],
        user_role=user.role,
        win_conditions=[]
    )
    
    agent = SecurityAgent(scenario)
    print("Ready to answer your questions.")
    
    # --- Main Loop ---
    while True:
        question = input("\nAsk a question (or type 'exit' to quit): ")
        
        if question.lower() == 'exit':
            break
            
        logging.info(f"Question: {question}")
        
        print("\nThinking...")
        
        # Use the SecurityAgent to get the answer
        result_generator = agent.ask(question)
        
        # Collect the full answer from the generator
        answer_parts = []
        for chunk in result_generator:
            if "output" in chunk:
                answer_parts.append(chunk["output"])
                print("\nAnswer:")
                print(chunk["output"])
                logging.info(f"Answer: {chunk['output']}")
                break  # For CLI, we just want the final output
        
        if not answer_parts:
            print("\nNo response received from agent.")
            logging.info("No response received from agent.")


    print("Session ended. Goodbye!")
    logging.info("User logged out.")


if __name__ == "__main__":
    main()
