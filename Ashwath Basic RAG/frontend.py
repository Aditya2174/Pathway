import streamlit as st
import os
from langchain_community.vectorstores.pathway import PathwayVectorClient    # Using the langchain client for Pathway vectorstores 
from langchain_core.documents import Document
import google.generativeai as genai


google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"  # Use your own API key pls
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8755
vector_client = PathwayVectorClient(host=PATHWAY_HOST, port=PATHWAY_PORT)

# Initializing chat history in Streamlit session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Pathway RAG Application")

# Document uploader - Main Vector Store....supports .txt and .json formats
st.subheader("Upload Document to Vector Store")
uploaded_file = st.file_uploader("Upload a document", type=["txt", "json"])
if uploaded_file:
    doc_text = uploaded_file.read().decode("utf-8")
    document = Document(page_content=doc_text)
    vector_client.add_documents([document])
    st.success("Document added to the vector store!")

# The chat input can also accept docuuments
st.subheader("Chat with Pathway Vector Store")
user_prompt = st.text_input("Enter your chat prompt:")
include_doc = st.checkbox("Include document in chat prompt")

attached_doc_text = None

if include_doc:
    attached_file = st.file_uploader("Attach document to prompt", type=["txt", "json"], key="attached_file")
    if attached_file:
        attached_doc_text = attached_file.read().decode("utf-8")
        with open(os.path.join("secondary_docs", attached_file.name), "w") as f:
            f.write(attached_doc_text)
        st.success("Document attached and saved.")

# Submit button logic
if st.button("Submit"):
    if user_prompt:
        # Retrieve context from the vector store
        retriever = vector_client.as_retriever()
        context = retriever.invoke(user_prompt)

        # Integrate the chat history into the prompt itself
        full_prompt = "\n\n".join(
            [f"User: {msg['user']}\nAI: {msg['ai']}" for msg in st.session_state.chat_history]
        )
        full_prompt += f"\n\nUser: {user_prompt}\nContext: {context}"
        if attached_doc_text:
            full_prompt += f"\n\nAttached Document: {attached_doc_text}"

        # Generate a response
        response = model.generate_content(full_prompt).text
        st.write("Response:", response)

        # Update chat history
        st.session_state.chat_history.append({"user": user_prompt, "ai": response})
    else:
        st.warning("Please enter a prompt for the chat!")
