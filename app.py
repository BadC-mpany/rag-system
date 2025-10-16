import streamlit as st
import os
from core.access_control import authenticate_user, load_users
from core.embeddings import get_embedding_model
from core.retrieval import run_rag_pipeline

# Initialize embedding model once
@st.cache_resource
def initialize_embeddings():
    return get_embedding_model()

def login_page():
    st.title("RAG System Login")
    
    users_data = load_users()
    usernames = [user.username for user in users_data]

    selected_username = st.selectbox("Select User", usernames)
    
    if st.button("Login as Selected User"):
        selected_user = next((user for user in users_data if user.username == selected_username), None)
        if selected_user:
            st.session_state["user"] = selected_user
            st.rerun()
        else:
            st.error("User not found.")

def chat_page():
    st.title("RAG System Chat")
    
    user = st.session_state["user"]
    st.write(f"Welcome, {user.username}! (Role: {user.role})")
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if prompt := st.chat_input("Ask a question"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                embeddings = initialize_embeddings()
                if embeddings:
                    
                    def stream_answer():
                        stream = run_rag_pipeline(user, prompt, embeddings, st.session_state.messages)
                        for chunk in stream:
                            if "answer" in chunk:
                                yield chunk["answer"]

                    response = st.write_stream(stream_answer)

                    # For now, let's get sources after streaming. A more advanced implementation
                    # could yield sources separately.
                    # This is a simplified approach to get sources after the answer is complete.
                    # We rerun the pipeline non-streamed to get the final context.
                    # NOTE: This is not efficient as it runs the query twice.
                    final_result = run_rag_pipeline(user, prompt, embeddings, st.session_state.messages)
                    sources = []
                    for chunk in final_result:
                        if "context" in chunk:
                            sources.extend(chunk["context"])

                    if sources:
                        st.markdown("---")
                        st.markdown("**Sources:**")
                        unique_sources = set(os.path.basename(doc.metadata.get('source', 'Unknown')) for doc in sources)
                        for source in unique_sources:
                            st.markdown(f"- {source}")
                else:
                    response = "Embedding model could not be initialized."
                    st.error(response)

            st.session_state.messages.append({"role": "assistant", "content": response})

def main():
    if "user" not in st.session_state:
        login_page()
    else:
        chat_page()

if __name__ == "__main__":
    main()
