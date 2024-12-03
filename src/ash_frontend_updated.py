# Streamlit for UI
import streamlit as st

# For file-related purposes
import os
import shutil
import diskcache as dc
import atexit
import json
import requests
import tempfile
import pdfplumber
import pandas as pd

# All LlamaIndex tools needed...LLM, memory, roles, etc
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from utils import process_user_query, get_colored_text
from llama_index.core.indices import VectorStoreIndex
from llama_index.core import Document
from llama_index.embeddings.google import GeminiEmbedding
import torch

# Autogen agents
from autogen import ConversableAgent, UserProxyAgent, register_function
from autogen.coding import LocalCommandLineCodeExecutor

# Summarizer of history
from transformers import pipeline

import requests

if not os.environ.get('AUTOGEN_USE_DOCKER'):
    os.environ['AUTOGEN_USE_DOCKER'] = '0'

google_api_key = "AIzaSyD939q3PbECaSJO1IAzRbmpqlREgJteLKg"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Initialize the WebSearchAPI
GOOGLE_CSE_ID = "AIzaSyBKTZOFWvwR-GMQvSEKthzOpDU86CB8Zoc"

tool_usage_log = {} # Dictionary to track tool usage

# Function to perform web search using Google Custom Search API
def web_search(query: str, api_key: str, cse_id: str, num_results: int = 5) -> str:
    """
    Perform a web search using Google Custom Search API.
    :param query: The search query string.
    :param api_key: Google API key.
    :param cse_id: Google Custom Search Engine ID.
    :param num_results: Number of search results to retrieve (default: 5).
    :return: A formatted string of search results or an error message.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        items = data.get('items', [])
        if items:
            return "\n".join([f"{item['title']} - {item['link']}" for item in items])
        return "No results found."
    except requests.exceptions.RequestException as e:
        return f"Error during web search: {e}"

# Function to log tool usage
def log_tool_usage(tool_name: str, query: str) -> None:
    """
    Log the usage of a tool.
    :param tool_name: Name of the tool being used.
    :param query: The query or input used with the tool.
    """
    if tool_name not in tool_usage_log:
        tool_usage_log[tool_name] = []
    tool_usage_log[tool_name].append(query)
    print(f"Tool '{tool_name}' used with query: {query}")

# Wrapper function with logging
def web_search_with_logging(query: str) -> str:
    """
    Perform a web search and log its usage.
    :param query: The search query string.
    :return: A formatted string of search results or an error message.
    """
    log_tool_usage("Web search tool", query)
    return web_search(query, api_key=google_api_key, cse_id=GOOGLE_CSE_ID)

# Constants/Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of messages after which chat history is summarized
SUMMARIZE_AFTER = 10
SMALL_FILE_SIZE_THRESHOLD = 5 * 1024 * 1024  # 5 MB

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8756

# Cache directory setup using diskcache
cache_dir = './document_cache'
document_cache = dc.Cache(cache_dir)

# Register cleanup function to delete cache on termination
def cleanup_cache():
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache directory '{cache_dir}' cleaned up.")

atexit.register(cleanup_cache)

temp_dir = tempfile.TemporaryDirectory()

# Initialize a Gemini-1.5-Flash model with LlamaIndex
gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)

retriever = PathwayRetriever(host=PATHWAY_HOST, port=PATHWAY_PORT)

# Initialize session variables
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'chat_messages_display' not in st.session_state:
    st.session_state.chat_messages_display = []
if 'summarized_history' not in st.session_state:
    st.session_state.summarized_history = ""
if 'message_counter' not in st.session_state:
    st.session_state.message_counter = 0
if 'summarizer' not in st.session_state:
    st.session_state.summarizer = pipeline("summarization", device=device, model="facebook/bart-large-cnn")
if 'df' not in st.session_state:
    st.session_state.df = None
if 'uploaded_docs' not in st.session_state:
    st.session_state.uploaded_docs = []
if 'uploaded_filenames' not in st.session_state:
        st.session_state.uploaded_filenames = set()
if 'sec_embedder' not in st.session_state:
    st.session_state.sec_embedder = GeminiEmbedding(model_name='models/embedding-001')
if 'sec_store' not in st.session_state:
    st.session_state.sec_store = VectorStoreIndex.from_documents(documents=st.session_state.uploaded_docs, **{'embed_model': st.session_state.sec_embedder})

agent_system_prompt = "Respond concisely and accurately, using the conversation provided and the context specified in the query. The user may reference documents they provided, which will be given to you as context.\
    You also have a web search tool and a code exeuction tool which can be used to retrieve real-time information or draw insights when necessary."
executor = LocalCommandLineCodeExecutor(work_dir=temp_dir.name)
auto_agent = ConversableAgent(name="assistant", human_input_mode="NEVER", system_message=agent_system_prompt,
                                llm_config={"config_list": [{"model": "gemini-1.5-flash", "temperature": 0.5, "api_key": os.environ.get("GOOGLE_API_KEY"), "api_type": "google"}]},
                                code_execution_config=False)

user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config={'executor': executor})

register_function(web_search_with_logging, caller=auto_agent, executor=user_proxy, name="search_tool", description="A tool to search the web and fetch information.")

def summarize_history(messages):
    history_text = " ".join([msg.content for msg in messages if msg.role != MessageRole.SYSTEM])

    # Use cached summarizer for processing
    summarizer = st.session_state.summarizer

    # No need to split if text is short
    if len(history_text) <= 1000:
        return summarizer(history_text, max_length=150, min_length=50, do_sample=False)[0]['summary_text']

    # Handle longer texts by splitting
    chunks = chunk_text(history_text, chunk_size=1000)  # Use an optimized chunk size
    summaries = [summarizer(chunk, max_length=150, min_length=50, do_sample=False)[0]['summary_text'] for chunk in chunks]
    return " ".join(summaries)

def chunk_text(text, chunk_size):
    """Split text into smaller chunks for parallel processing."""
    words = text.split()
    return [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

def extract_non_table_text(page):
    non_table_text = ""

    table_bboxes = [table.bbox for table in page.find_tables()]

    for word in page.extract_words():
        word_bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
        if not any((word_bbox[0] >= bbox[0] and word_bbox[1] >= bbox[1] and
                    word_bbox[2] <= bbox[2] and word_bbox[3] <= bbox[3]) for bbox in table_bboxes):
            non_table_text += word['text'] + " "

    return non_table_text.strip()

def read_file_from_cache_or_parse(uploaded_file, cache, file_size_threshold):
    """Reads a file from cache if available, otherwise parses it."""
    file_name = uploaded_file.name
    file_type = uploaded_file.type

    if file_name in cache:
        st.success(f"Loaded cached document: {file_name}")
        return cache[file_name]

    attached_text = ""

    if file_type == "application/pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                non_table_text = extract_non_table_text(page)
                attached_text += non_table_text + "\n"

                # Extract tables
                tables = page.extract_tables()
                if tables:
                    attached_text += "\n--- Table Data ---\n"
                    for table in tables:
                        for row in table:
                            attached_text += " | ".join(cell if cell else "" for cell in row) + "\n"
                    attached_text += "--- End of Table ---\n"

    elif file_type == "text/plain":
        attached_text = uploaded_file.read().decode("utf-8")

    elif file_type == "application/json":
        attached_text = json.dumps(json.load(uploaded_file), indent=2)

    elif file_type == "text/csv":
        if uploaded_file.size <= file_size_threshold:
            df = pd.read_csv(uploaded_file)
            attached_text = df.to_string(index=False)
        else:
            df = pd.read_csv(uploaded_file)
            summary_info = f"""
            File Size: {uploaded_file.size / (1024 * 1024):.2f} MB
            Number of Rows: {df.shape[0]}
            Number of Columns: {df.shape[1]}
            Columns: {', '.join(df.columns)}
            """
            attached_text = summary_info
            st.write("Preview of Data:")
            st.dataframe(df.head(10))

    cache[file_name] = attached_text
    return attached_text

# Document uploader for including document text in the chat prompt
with st.sidebar:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more documents (.txt, .json, .csv, .pdf)", 
        type=["txt", "json", "pdf", "csv"], 
        accept_multiple_files=True
    )
    combined_attached_text = ""

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            if file_name in st.session_state.uploaded_filenames:
                st.warning(f"'{file_name}' has already been added to the vector store.")
                combined_attached_text += st.session_state.document_cache[file_name] + "\n"
            else:
                attached_text = read_file_from_cache_or_parse(
                    uploaded_file,
                    st.session_state.document_cache,
                    SMALL_FILE_SIZE_THRESHOLD
                )
                combined_attached_text += attached_text + "\n"

                # Add the parsed text as a Document for the vector store
                doc = Document(text=attached_text, metadata={"filename": file_name})
                st.session_state.uploaded_docs.append(doc)
                st.session_state.sec_store.insert(doc)
                st.session_state.uploaded_filenames.add(file_name)

                st.success(f"Document '{file_name}' added to the secondary vector store!")

# Display the chat history
for chat in st.session_state.chat_messages_display:
    role = "assistant" if chat.role == MessageRole.ASSISTANT else "user"
    with st.chat_message(role):
        st.markdown(chat.content, unsafe_allow_html=True)

# Chat input for user to enter prompt
if user_input := st.chat_input("Enter your chat prompt:"):
    st.session_state.message_counter += 1
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        # Summarize history periodically
        if st.session_state.message_counter % SUMMARIZE_AFTER == 0:
            with st.spinner(text="Summarizing history..."):
                st.session_state.summarized_history = summarize_history(st.session_state.chat_messages)
                st.session_state.chat_messages = [
                    st.session_state.chat_messages[0],
                    ChatMessage(role=MessageRole.ASSISTANT, content=st.session_state.summarized_history)
                ]
                st.success("Chat history summarized!")

        full_prompt = f"{user_input}\n\nAttached Document: {combined_attached_text}" if combined_attached_text else user_input
        with st.spinner("Analyzing user query..."):
            # contextualized_prompt = contextualize_prompt(user_input)
            document_txt = combined_attached_text if combined_attached_text else ""
            query_type, preprocess_op = process_user_query(gemini_model, st.session_state.chat_messages, user_input, document=document_txt)

        if 'general' in query_type:
            # Provide response directly for 'general' queries
            print('Query Type: General')
            st.markdown(preprocess_op)
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
            st.session_state.chat_messages_display.append(ChatMessage(role=MessageRole.USER, content=user_input))
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=preprocess_op))
            st.session_state.chat_messages_display.append(ChatMessage(role=MessageRole.ASSISTANT, content=preprocess_op))
        else:
            green_txt_reform = get_colored_text(preprocess_op)
            st.markdown(green_txt_reform, unsafe_allow_html=True)

            display_msg = user_input + "<br>" + green_txt_reform

            st.session_state.chat_messages_display.append(ChatMessage(role=MessageRole.USER, content=display_msg))
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=preprocess_op))
            if 'direct' in query_type:
                print("Query Type: Direct")
                
                if combined_attached_text:
                    # Modify the last message in the session state to include document context
                    last_message = st.session_state.chat_messages[-1]
                    last_message.content = f"{last_message.content}\n\nAttached Document Context:\n{combined_attached_text}"

                formatted_messages = [
                    {"role": "user" if msg.role == MessageRole.USER else "assistant", "content": msg.content}
                    for msg in st.session_state.chat_messages
                ]
                auto_reply = auto_agent.generate_reply(messages=formatted_messages)

                chat_result = user_proxy.initiate_chat(recipient=auto_agent, message=st.session_state.chat_messages[-1].content)
                print(f"Chat Result:\n{chat_result['chat_history'][-1]['content']}")
                st.markdown(auto_reply['content'])
                st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=auto_reply['content']))
                st.session_state.chat_messages_display.append(ChatMessage(role=MessageRole.ASSISTANT, content=auto_reply['content']))

            elif 'context' in query_type:
                print("Query Type: Context")
                system_prompt = "Respond concisely and accurately, using the conversation provided and the context specified in the query as context."
                contextualized_prompt = preprocess_op
                # Retrieve additional context
                context = retriever.retrieve(contextualized_prompt)
                context_text = "\n".join([doc.text for doc in context])

                sec_retriever = st.session_state.sec_store.as_retriever()
                context = sec_retriever.retrieve(contextualized_prompt)
                sec_context_text = "\n".join([doc.text for doc in context])

                combined_context_text = f"Context from database:\n{context_text}\n\nContext from document uploaded by user:\n{sec_context_text}"

                # Prepare the input for the ConversableAgent
                formatted_messages = [
                    {"role": "user" if msg.role == MessageRole.USER else "assistant", "content": msg.content}
                    for msg in st.session_state.chat_messages[:-1]
                ]

                # Include context and the reformulated query
                formatted_messages.append({"role": "user", "content": f"{contextualized_prompt}\nContext:\n{context_text}"})

                # Generate a response using the ConversableAgent
                response = auto_agent.generate_reply(messages=formatted_messages)
                chat_result = user_proxy.initiate_chat(recipient=auto_agent, message=st.session_state.chat_messages[-1].content)
                print(f"Chat Result:\n{chat_result}")
                assistant_response = response['content']

                # Display the response
                st.markdown(assistant_response)
                st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response))
                st.session_state.chat_messages_display.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response))
            else:
                print("Invalid response type detected:", query_type)
                st.error("Invalid response type detected. Please try again.")

    print(st.session_state.chat_messages)
