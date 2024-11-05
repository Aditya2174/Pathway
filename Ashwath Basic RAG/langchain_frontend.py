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
    template = [
        SystemMessage(content="Analyze if the prompt is a follow-up question or a standalone question. "
                              "If it's a follow-up that needs prior context, respond with 'needs more context'. "
                              "If the prompt can be answered directly, respond with 'direct answer'. "
                              "If it's an independent question, respond with 'standalone question'."),
        MessagesPlaceholder("chat_history"),
        HumanMessage(content=user_prompt)
    ]
    
    final_prompt = ChatPromptTemplate.from_messages(template).format_prompt(**{"chat_history": st.session_state.memory.chat_memory.messages})
    response = model.invoke(final_prompt).content.lower()

    if 'direct answer' in response:
        return "direct_answer"
    elif 'needs more context' in response:
        return "needs_more_context"
    else:
        return "standalone_question"

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
        response_type = can_answer(user_prompt, attached_doc_text or "")

        if "direct answer" in response_type.lower():
            st.write("Direct answer")
            full_prompt = user_prompt
            if attached_doc_text:
                full_prompt += f"\n\nAttached Document: {attached_doc_text}"

            template = [
                SystemMessage(content="Answer the user’s prompt based on the chat history and the attached document (if any)."),
                MessagesPlaceholder("chat_history"),
                HumanMessage(content=full_prompt)
            ]
            chat_prompt_template = ChatPromptTemplate.from_messages(template)

            # Format the prompt and get the response
            formatted_prompt = chat_prompt_template.format_prompt(**{"chat_history": st.session_state.memory.chat_memory.messages})
            response = model.invoke(formatted_prompt).content
            final_response = agent.stream(formatted_prompt)
            stream_responses = [message for message in final_response]

            response_content = stream_responses[-1]['agent']['messages'][0].content
            st.write(f"Final Response: {response_content}")

            # Update memory with the new user message and context
            st.session_state.memory.chat_memory.add_message(HumanMessage(content=user_prompt))
            st.session_state.memory.chat_memory.add_message(AIMessage(content=response_content))

        elif "needs_more_context" in response_type.lower():
            st.write("Needs more context")
            retriever = vector_client.as_retriever()
            context = retriever.invoke(user_prompt)
            
            content_list = [doc.page_content for doc in context]
            st.write(f"Context: {content_list}")

            full_prompt = f"{user_prompt}\n\nContext: {content_list}"
            if attached_doc_text:
                full_prompt += f"\n\nAttached Document: {attached_doc_text}"

            template = [
                SystemMessage(content="Answer using the context provided."),
                MessagesPlaceholder("chat_history"),
                HumanMessage(content=full_prompt)
            ]
            chat_prompt_template = ChatPromptTemplate.from_messages(template)

            formatted_prompt = chat_prompt_template.format_prompt(**{"chat_history": st.session_state.memory.chat_memory.messages})
            final_response = agent.stream(formatted_prompt)
            stream_responses = [message for message in final_response]

            response_content = stream_responses[-1]['agent']['messages'][0].content
            st.write(f"Final Response: {response_content}")

            st.session_state.memory.chat_memory.add_message(HumanMessage(content=user_prompt))
            st.session_state.memory.chat_memory.add_message(AIMessage(content=response_content))

        else:  # standalone_question
            st.write("Standalone question")
            full_prompt = user_prompt
            if attached_doc_text:
                full_prompt += f"\n\nAttached Document: {attached_doc_text}"

            template = [
                SystemMessage(content="Answer the user’s prompt as a standalone question."),
                HumanMessage(content=full_prompt)
            ]
            chat_prompt_template = ChatPromptTemplate.from_messages(template)

            # Format the prompt and get the response
            formatted_prompt = chat_prompt_template.format_prompt(**{"chat_history": st.session_state.memory.chat_memory.messages})
            final_response = agent.stream(formatted_prompt)
            stream_responses = [message for message in final_response]

            response_content = stream_responses[-1]['agent']['messages'][0].content
            st.write(f"Final Response: {response_content}")

            st.session_state.memory.chat_memory.add_message(HumanMessage(content=user_prompt))
            st.session_state.memory.chat_memory.add_message(AIMessage(content=response_content))

    else:
        st.warning("Please enter a prompt for the chat!")
