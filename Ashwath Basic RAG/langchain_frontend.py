import streamlit as st
import os
from langchain_community.vectorstores.pathway import PathwayVectorClient
from langchain_core.documents import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
import json

# Set up Google Gemini model with LangChain
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"  # Remember to replace with your API key
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8755
vector_client = PathwayVectorClient(host=PATHWAY_HOST, port=PATHWAY_PORT)

# Initialize memory for chat history
memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_prompt", return_messages=True)

st.title("Pathway RAG Application")

# Document uploader - Main Vector Store
st.subheader("Upload Document to Vector Store")
uploaded_file = st.file_uploader("Upload a document", type=["txt", "json"])
if uploaded_file:
    doc_text = uploaded_file.read().decode("utf-8")
    document = Document(page_content=doc_text)
    vector_client.add_documents([document])
    st.success("Document added to the vector store!")

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

        # Prepare full prompt with context and attached document
        full_prompt = f"{user_prompt}\n\nContext: {context}"
        if attached_doc_text:
            full_prompt += f"\n\nAttached Document: {attached_doc_text}"

        # Update memory with the new user message and context
        memory.chat_memory.add_message(HumanMessage(content=user_prompt))
        
        # Create chat prompt template with memory
        template = [
            SystemMessage(content="Answer using the context provided."),
            MessagesPlaceholder("chat_history"),
            HumanMessage(content=full_prompt)
        ]
        chat_prompt_template = ChatPromptTemplate.from_messages(template)
        
        # Format the prompt and get the response
        formatted_prompt = chat_prompt_template.format_prompt(**{"chat_history": memory.chat_memory.messages})
        response = model.invoke(formatted_prompt).content
        st.write("Response:", response)

        # Save AI response back to memory
        memory.chat_memory.add_message(AIMessage(content=response))
    else:
        st.warning("Please enter a prompt for the chat!")
