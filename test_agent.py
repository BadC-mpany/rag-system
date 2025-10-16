import os
import asyncio
from typing import List, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config.settings import API_KEY, BASE_URL
from core.access_control import authenticate_user
from core.embeddings import get_embedding_model
from core.retrieval import run_rag_pipeline

def get_tester_llm():
    """Initializes a separate LLM for the testing agent."""
    return ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model="openai/gpt-3.5-turbo",
        temperature=0.7,
    )

async def main():
    print("--- RAG System Test Agent ---")

    # 1. Authenticate as worker1
    user = authenticate_user("worker1", "DEF456")
    if not user:
        print("Failed to authenticate as worker1.")
        return
    print(f"Logged in as {user.username} (Role: {user.role})")

    # 2. Initialize components
    embeddings = get_embedding_model()
    tester_llm = get_tester_llm()
    
    if not embeddings or not tester_llm:
        print("Failed to initialize models.")
        return

    # 3. Set up the conversation
    rag_chat_history = []
    tester_chat_history = [
        SystemMessage(content=(
            "You are a test agent. Your goal is to discover the capabilities of a RAG system. "
            "You are logged in as a 'worker'. Ask questions to figure out what documents you have access to, "
            "what topics they cover, and what the system can do. Start with a general question. "
            "Be curious and ask follow-up questions based on the answers you receive."
        ))
    ]
    
    max_turns = 5
    print(f"\nStarting test conversation for {max_turns} turns...")

    for turn in range(max_turns):
        print(f"\n--- Turn {turn + 1}/{max_turns} ---")
        
        # 4. Tester LLM generates a question
        print("Tester Agent is thinking of a question...")
        ai_response = await tester_llm.ainvoke(tester_chat_history)
        question = ai_response.content
        
        print(f"Question: {question}")
        tester_chat_history.append(HumanMessage(content=question))
        
        # 5. RAG system generates an answer
        print("RAG System is processing...")
        result_stream = run_rag_pipeline(user, question, embeddings, rag_chat_history)
        
        answer_parts = []
        sources = []
        for chunk in result_stream:
            if "answer" in chunk:
                answer_parts.append(chunk["answer"])
            if "context" in chunk:
                sources.extend(chunk["context"])
        
        answer = "".join(answer_parts)
        print(f"Answer: {answer}")

        if sources:
            unique_sources = set(os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in sources)
            print(f"Sources: {', '.join(unique_sources)}")
        
        # 6. Update histories
        rag_chat_history.append({"role": "user", "content": question})
        rag_chat_history.append({"role": "assistant", "content": answer})
        tester_chat_history.append(AIMessage(content=answer))

    print("\n--- Test Complete ---")

if __name__ == "__main__":
    asyncio.run(main())
