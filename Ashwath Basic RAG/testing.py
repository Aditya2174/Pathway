# Necessary imports
import streamlit as st
import os
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec

# Initialize a Gemini-1.5-Flash model with LlamaIndex
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8755

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

# Initializing a simple agent with a search tool
agent = ReActAgent.from_tools(tools=search_tool, llm=gemini_model)

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Answer the question with whatever information is available, and if more is needed, use the tool for a web search.\
                     Use the context provided if any is available. "
                "If there is not enough information still, ask the user for clarification."
            ),
        ),
    ]

# Function to reformat user's prompt with history as context
def contextualize_prompt(user_prompt):
    context_template = [ChatMessage(role=MessageRole.SYSTEM, content="Use the chat history to recontextualize the user's prompt. \
        Remove ambiguous references and prepare it for answering. \
        Return the re-contextualized prompt.")]
    context_template.extend(st.session_state.chat_messages[1:])
    context_template.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

    reformat_prompt = ChatPromptTemplate.from_messages(context_template).format_messages()
    response = gemini_model.chat(messages=reformat_prompt).message.content

    return response

# Identifying whether retrieval is needed
def determine_response_type(user_prompt):
    analysis_template = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Analyze if the prompt can be answered using chat history alone or if it requires document retrieval or a web search."
                "If chat history is sufficient, respond with 'direct answer'."
                "If additional documents or web searches are needed, respond with 'needs more context'."
            ),
        ),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]
    analysis_template = ChatPromptTemplate.from_messages(analysis_template).format_messages()
    analysis_template.extend(st.session_state.chat_messages[1:])
    response = gemini_model.chat(analysis_template).message.content.lower()

    if 'direct answer' in response:
        return "direct_answer"
    elif 'needs more context' in response:
        return "needs_more_context"

# Document uploader for including document text in the chat prompt
with st.sidebar:
    st.subheader("Upload Document Text for Context")
    uploaded_file = st.file_uploader("Upload a document", type=["txt", "json"])
    attached_doc_text = None
    if uploaded_file:
        attached_doc_text = uploaded_file.read().decode("utf-8")
        st.success("Document text loaded for context!")

# Display the chat history
for chat in st.session_state.chat_messages:
    role = "assistant" if chat.role == MessageRole.ASSISTANT else "user"
    st.chat_message(role).write(chat.content)

# Chat input for user to enter prompt
if user_input := st.chat_input("Enter your chat prompt:"):
    # Add the user's message to chat history immediately
    st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
    st.chat_message("user").write(user_input)

    # Include document text in the prompt if checked
    full_prompt = f"{user_input}\n\nAttached Document: {attached_doc_text}" if attached_doc_text else user_input

    # Process user prompt
    contextualized_prompt = contextualize_prompt(full_prompt)
    response_type = determine_response_type(contextualized_prompt)
    
    # Retrieve answer based on response type
    if response_type == "direct_answer":
        context = retriever.retrieve(contextualized_prompt)
        context_text = "\n".join([doc.text for doc in context])
        final_prompt = f"Prompt: {contextualized_prompt}\nContext: {context_text}"
        
        # Add the user's message to chat history
        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=final_prompt))
        
        # Get the model's response
        response = gemini_model.chat(ChatPromptTemplate.from_messages(st.session_state.chat_messages).format_messages()).message.content
        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))
        st.chat_message("assistant").write(response)

    elif response_type == "needs_more_context":
        context = retriever.retrieve(contextualized_prompt)
        context_text = "\n".join([doc.text for doc in context])
        final_prompt = f"Prompt: {contextualized_prompt}\nContext: {context_text}"
        
        # Add the user's message to chat history
        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=final_prompt))
        
        # Get the agent's response
        response = agent.chat(message=final_prompt, chat_history=st.session_state.chat_messages).response
        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))
        st.chat_message("assistant").write(response)
