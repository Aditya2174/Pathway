import streamlit as st
import os
from langchain_community.vectorstores.pathway import PathwayVectorClient
from langchain_core.documents import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

# Set up Google Gemini model with LangChain
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"  # Remember to replace with your API key
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8755
vector_client = PathwayVectorClient(host=PATHWAY_HOST, port=PATHWAY_PORT)

# Use session state to initialize memory
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_prompt", return_messages=True)

# Creating a search tool and configuring API settings
tavily_api_key = "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"
if not os.environ.get('TAVILY_API_KEY'):
    os.environ['TAVILY_API_KEY'] = tavily_api_key

search = TavilySearchResults(max_results=5, include_answer=True)
tools = [search]

# Creating a simple agent to handle tool calling
agent = create_react_agent(model=model, tools=tools)

def can_answer(user_prompt, attached_doc_text):
    # Get recent context from memory
    history_text = "\n".join([msg.content for msg in st.session_state.memory.chat_memory.messages])
    full_prompt = f"{history_text}\n\nUser Prompt: {user_prompt}"

    if attached_doc_text:
        full_prompt += f"\n\nAttached Document: {attached_doc_text}"

    template = [
        SystemMessage(content=( 
            "Determine if you can answer the user’s prompt based on the current chat history and any attached document. "
            "If the chat history or attached document provides enough context to answer, respond with 'yes'. "
            "If they do not provide enough context, respond with 'no', and do not attempt to answer the question. "
        )),
        MessagesPlaceholder("chat_history"),
        HumanMessage(content=full_prompt)
    ]

    final_prompt = ChatPromptTemplate.from_messages(template).format_prompt(**{"chat_history": st.session_state.memory.chat_memory.messages})
    response = model.invoke(final_prompt).content

    return 1 if 'yes' in response.lower() else 0

st.title("Pathway RAG Application")

# Document uploader - Main Vector Store
with st.sidebar:
    st.subheader("Upload Document to Vector Store")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "json"])
    if uploaded_file:
        doc_text = uploaded_file.read().decode("utf-8")
        document = Document(page_content=doc_text)
        vector_client.add_documents([document])
        st.success("Document added to the vector store!")

# Chat input with document option
st.subheader("Chat with Pathway Vector Store")
user_prompt = st.text_input("Enter your chat prompt:")
include_doc = st.checkbox("Include document in chat prompt")

# Extracting text from attached document (if provided)
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
        # Check if the history and attached document (if any) are sufficient to answer
        is_sufficient = can_answer(user_prompt, attached_doc_text or "")

        # If sufficient, proceed to answer based on the document and chat history
        if is_sufficient:
            full_prompt = user_prompt
            if attached_doc_text:
                full_prompt += f"\n\nAttached Document: {attached_doc_text}"

            template = [
                SystemMessage(content="Answer the user’s prompt based on the chat history and the attached document(if any)."),
                MessagesPlaceholder("chat_history"),
                HumanMessage(content=full_prompt)
            ]
            chat_prompt_template = ChatPromptTemplate.from_messages(template)

            # Format the prompt and get the response
            formatted_prompt = chat_prompt_template.format_prompt(**{"chat_history": st.session_state.memory.chat_memory.messages})
            response = model.invoke(formatted_prompt).content
            st.write(f"Final Response: {response}")

            # Update memory with the new user message and context
            st.session_state.memory.chat_memory.add_message(HumanMessage(content=user_prompt))
            # Save AI response back to memory
            st.session_state.memory.chat_memory.add_message(AIMessage(content=response))

        # If not sufficient, retrieve context from the vector store and respond
        else:
            # Retrieve context from the vector store        
            retriever = vector_client.as_retriever(search_type="similarity_score_threshold", search_kwargs={'score_threshold': 0.5})
            context = retriever.invoke(user_prompt)

            # Prepare full prompt with context and attached document
            full_prompt = f"{user_prompt}\n\nContext: {context}"
            if attached_doc_text:
                full_prompt += f"\n\nAttached Document: {attached_doc_text}"

            # Create chat prompt template with memory
            template = [
                SystemMessage(content="Answer using the context provided."),
                MessagesPlaceholder("chat_history"),
                HumanMessage(content=full_prompt)
            ]
            chat_prompt_template = ChatPromptTemplate.from_messages(template)

            # Format the prompt and get the response
            formatted_prompt = chat_prompt_template.format_prompt(**{"chat_history": st.session_state.memory.chat_memory.messages})
            final_response = agent.stream(formatted_prompt)
            stream_responses = [message for message in final_response]

            response_content = stream_responses[-1]['agent']['messages'][0].content
            st.write(f"Final Response: {response_content}")

            # Update memory with the new user message and context
            st.session_state.memory.chat_memory.add_message(HumanMessage(content=user_prompt))
            # Save AI response back to memory
            st.session_state.memory.chat_memory.add_message(AIMessage(content=response_content))

    else:
        st.warning("Please enter a prompt for the chat!")
