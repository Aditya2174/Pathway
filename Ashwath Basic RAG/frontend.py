import streamlit as st
import os
import json
from langchain_community.vectorstores.pathway import PathwayVectorClient  # Using the langchain client for Pathway vectorstores 
from langchain_core.documents import Document
import google.generativeai as genai
from server_runner import vector_store_server

# Configure Google API key
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"  # Use your own API key
genai.configure(api_key=google_api_key)
model = genai.GenerativeModel('gemini-1.5-flash')
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8755
vector_client = PathwayVectorClient(host=PATHWAY_HOST, port=PATHWAY_PORT)

# Create the data directory if it doesn't exist
DATA_DIRECTORY = "data"
os.makedirs(DATA_DIRECTORY, exist_ok=True)

# Initialize chat history in Streamlit session
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

st.title("Pathway RAG Application")

# Document uploader - Save to Data Directory for Vector Store Sync
st.subheader("Upload Document to Vector Store")
uploaded_file = st.file_uploader("Upload a document", type=["txt", "json"])
if uploaded_file:
    doc_text = uploaded_file.read().decode("utf-8")
    document_path = os.path.join(DATA_DIRECTORY, uploaded_file.name)
    with open(document_path, "w") as f:
        f.write(doc_text)
    st.success("Document saved to the data directory! Pathway server will index it shortly.")

# Chat Input with Document Option
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
        
        fused_scores = {}
        k=60
        for docs in context:
            for rank, doc in enumerate(docs):
                doc_str = json.dumps(doc)
                # If the document is not yet in the fused_scores dictionary, add it with an initial score of 0
                # print('\n')
                if doc_str not in fused_scores:
                    fused_scores[doc_str] = 0
                    # Retrieve the current score of the document, if any
                    previous_score = fused_scores[doc_str]
                    # Update the score of the document using the RRF formula: 1 / (rank + k)
                    fused_scores[doc_str] += 1 / (rank + k)

        # final reranked result
        reranked_results = [
            (json.loads(doc), score)
            for doc, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)
        ]


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

