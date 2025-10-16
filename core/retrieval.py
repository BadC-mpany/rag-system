from typing import List, Dict, Any

from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

from core.llm import get_llm
from core.vectorstore import create_or_load_vectorstore
from users.schema import User
from utils.file_io import load_documents_from_directories
from core.access_control import get_accessible_directories

def run_rag_pipeline(user: User, question: str, embeddings, chat_history: List[Dict[str, str]]) -> Dict[str, Any]:
    """
    Runs the RAG pipeline for a given user, question, and chat history.
    """
    accessible_dirs = get_accessible_directories(user)
    documents = load_documents_from_directories(accessible_dirs)

    if not documents:
        return {"answer": "No documents accessible to the user.", "source_documents": []}

    vector_store = create_or_load_vectorstore(documents, embeddings, user.role)
    retriever = vector_store.as_retriever()

    llm = get_llm()
    if not llm:
        return {"answer": "LLM could not be initialized.", "source_documents": []}

    # Contextualize question prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    # Answering prompt
    qa_system_prompt = (
        "You are InnovateX's Corporate Information Assistant, a friendly and accurate AI. "
        "Your primary role is to provide precise answers based *only* on the retrieved context, "
        "acting as a helpful and engaging conversational partner. "
        "Always start by greeting the user and offering help. "
        "If the context does not contain the answer, politely state that you don't have enough information. "
        "Ensure your responses are concise, clear, and relevant to the user's query, always maintaining a professional and helpful tone. "
        "Use markdown for formatting when appropriate (e.g., bullet points for lists, bold for emphasis)."
        "\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    # Convert chat history format
    langchain_chat_history = []
    for message in chat_history:
        if message["role"] == "user":
            langchain_chat_history.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            langchain_chat_history.append(AIMessage(content=message["content"]))

    try:
        return rag_chain.stream({"input": question, "chat_history": langchain_chat_history})
    except Exception as e:
        print(f"Error during RAG pipeline execution: {e}")
        return iter([{"answer": "An error occurred while generating the answer.", "context": []}])
