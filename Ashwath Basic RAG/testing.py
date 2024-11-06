import streamlit as st
import os
import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreClient
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.tools.function_tool import FunctionTool

# Initialize Google Gemini model with LlamaIndex
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"  # Replace with your actual API key
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8755
vector_client = VectorStoreClient(host=PATHWAY_HOST, port=PATHWAY_PORT)
retriever = PathwayRetriever(host=PATHWAY_HOST, port=PATHWAY_PORT)

# Initialize memory if not already in session state
if 'memory' not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer(token_limit=10000)

# Tavily search tool setup
tavily_api_key = "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"
if not os.environ.get('TAVILY_API_KEY'):
    os.environ['TAVILY_API_KEY'] = tavily_api_key
search_tool = TavilyToolSpec(api_key=tavily_api_key)
search_tool = search_tool.to_tool_list()

agent = ReActAgent.from_tools(tools=search_tool, llm=gemini_model)

chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Always answer the question, even if the context isn't helpful."
            ),
        ),
]
text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs)

# Streamlit UI components
st.title("Pathway RAG Application with Gemini")

# Document uploader for including document text in the chat prompt
with st.sidebar:
    st.subheader("Upload Document Text for Context")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "json"])
    attached_doc_text = None
    if uploaded_file:
        attached_doc_text = uploaded_file.read().decode("utf-8")
        st.success("Document text loaded for context!")

# Chat input with optional document attachment
st.subheader("Chat with Pathway Vector Store")
user_prompt = st.text_input("Enter your chat prompt:")
include_doc = st.checkbox("Include uploaded document text in chat prompt")

# Append document text to prompt if required
if include_doc and attached_doc_text:
    full_prompt = f"{user_prompt}\n\nAttached Document: {attached_doc_text}"
else:
    full_prompt = user_prompt

# Logic to determine response type and generate answer
def determine_response_type(user_prompt):
    chat_text_qa_msgs = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Analyze if the prompt is a follow-up question or a standalone question. "
                "If it's a follow-up that needs prior context, respond with 'needs more context'. "
                "If the prompt can be answered using information provided or with chat history respond with 'direct answer'. "
            ),
        ),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]
    text_qa_template = ChatPromptTemplate.from_messages(chat_text_qa_msgs).format_messages()

    response = gemini_model.chat(text_qa_template).message.content.lower()
    st.write("Response type: ", response)

    if 'direct answer' in response:
        return "direct_answer"
    elif 'needs more context' in response:
        return "needs_more_context"

# Submit button logic
if st.button("Submit"):
    if user_prompt:
        response_type = determine_response_type(user_prompt)
        
        if response_type == "direct_answer":
            st.write("Direct answer")
            context = retriever.retrieve(user_prompt)
            context_text = " ".join([doc.text for doc in context])
            full_prompt = f"Prompt: {full_prompt}\n Context: {context_text}"
            chat_text_qa_msgs.append(ChatMessage(role=MessageRole.USER, content=full_prompt))
            final_prompt = ChatPromptTemplate.from_messages(chat_text_qa_msgs).format_messages()
            response = gemini_model.chat(final_prompt).message.content
            chat_text_qa_msgs.append(ChatMessage(role=MessageRole.CHATBOT, content=response))
            st.write(f"Final Answer: {response}")
        elif response_type == "needs_more_context":
            st.write("Needs more context")
            context = retriever.retrieve(user_prompt)
            context_text = " ".join([doc.text for doc in context])
            full_prompt = f"Prompt: {full_prompt}\n Context: {context_text}"
            
            
            response = agent.chat(message=full_prompt, chat_history=chat_text_qa_msgs)
            # final_prompt = ChatPromptTemplate.from_messages(chat_text_qa_msgs).format_messages()
            # response = gemini_model.chat(final_prompt).message.content
            chat_text_qa_msgs.append(ChatMessage(role=MessageRole.USER, content=full_prompt))
            chat_text_qa_msgs.append(ChatMessage(role=MessageRole.CHATBOT, content=response))
            st.write(f"Final Answer: {response}")
    else:
        st.warning("Please enter a prompt for the chat!")
