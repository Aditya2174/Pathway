# Streamlit for UI
import streamlit as st

# For file-related purposes
import os
import time
import shutil
from typing import Annotated
import diskcache as dc
import atexit
import json
import requests
import pdfplumber
import pandas as pd

# All LlamaIndex tools needed...LLM, memory, roles, etc
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from utils import process_user_query, get_colored_text, get_history_str
from llama_index.core.indices import VectorStoreIndex
from llama_index.core import Document
from llama_index.embeddings.google import GeminiEmbedding
import torch

# Autogen agents
from autogen import ConversableAgent, UserProxyAgent, register_function
from autogen.coding import LocalCommandLineCodeExecutor

# Summarizer of history
from transformers import pipeline
from typing import Tuple
import requests
from tavily import TavilyClient

# Tavily search tool setup
tavily_api_key = "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"
if not os.environ.get('TAVILY_API_KEY'):
    os.environ['TAVILY_API_KEY'] = tavily_api_key

tavily = TavilyClient(tavily_api_key)

def search_tool(query: Annotated[str, "The search query"]) -> Annotated[str, "The search results"]:
    return tavily.get_search_context(query=query, search_depth="advanced")

def evaluate_sufficiency(context_text: str, query: str, total_cost :int) -> bool:
        """Use Gemini model to evaluate if the retrieved context suffices to answer the query."""
        evaluation_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context_text}\n\n"
            "Does the context provide sufficient information to fully answer the question?"
            "Respond with 'Yes' or 'No' only."
        )
        time.sleep(1)
        # result = gemini_model.generate_response(evaluation_prompt)
        result = gemini_model.chat([ChatMessage(content=evaluation_prompt, role=MessageRole.USER)])
        cost = result.raw['usage_metadata']['total_token_count']
        total_cost+= cost
        if 'no' in result.message.content.lower():
            return False
        return True

if not os.environ.get('AUTOGEN_USE_DOCKER'):
    os.environ['AUTOGEN_USE_DOCKER'] = '0'

google_api_key = "AIzaSyD939q3PbECaSJO1IAzRbmpqlREgJteLKg"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Initialize the WebSearchAPI
# GOOGLE_CSE_ID = "AIzaSyBKTZOFWvwR-GMQvSEKthzOpDU86CB8Zoc"
google_cse_id = "d3a75edbda16e451e"

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
    # url = "https://www.googleapis.com/customsearch/v1"
    url = "https://cse.google.com/cse"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        print(f"Search response: {response}")
        print(f"Response content: {response.content}")
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
    return web_search(query, api_key=google_api_key, cse_id=google_cse_id)

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

os.makedirs("coding", exist_ok=True)    # Create a working directory for code executor

gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)  # Initialize a Gemini-1.5-Flash model with LlamaIndex

retriever = PathwayRetriever(host=PATHWAY_HOST, port=PATHWAY_PORT, similarity_top_k=5)

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
    You also have a web search tool and a code exeuction tool which can be used to retrieve real-time information or draw insights when necessary.\
        If extra information is needed to answer the question, use a web search."
executor = LocalCommandLineCodeExecutor(work_dir="coding", timeout=15)
auto_agent = ConversableAgent(name="assistant", human_input_mode="NEVER", system_message=agent_system_prompt,
                                llm_config={"config_list": [{"model": "gemini-1.5-flash", "temperature": 0.5, "api_key": os.environ.get("GOOGLE_API_KEY"), "api_type": "google"}]},
                                code_execution_config=False)

user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config={'executor': executor},
                            default_auto_reply="If you have any more important information to add, add it. Otheriwse, respond with 'done'")

# register_function(web_search_with_logging, caller=auto_agent, executor=user_proxy, name="search_tool", description="A tool to search the web and fetch information.")

register_function(
    search_tool,
    caller=auto_agent,
    executor=user_proxy,
    name="search_tool",
    description="Search the web for the given query",
)

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
combined_attached_text = ""
# Document uploader for including document text in the chat prompt
with st.sidebar:
    st.subheader("Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload one or more documents (.txt, .json, .csv, .pdf)", 
        type=["txt", "json", "pdf", "csv"], 
        accept_multiple_files=True
    )
    

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            if file_name in st.session_state.uploaded_filenames:
                st.warning(f"'{file_name}' has already been added to the vector store.")
                combined_attached_text += document_cache[file_name] + "\n"
            else:
                attached_text = read_file_from_cache_or_parse(
                    uploaded_file,
                    document_cache,
                    SMALL_FILE_SIZE_THRESHOLD
                )
                combined_attached_text += attached_text + "\n"

                # Add the parsed text as a Document for the vector store
                doc = Document(text=attached_text, metadata={"filename": file_name})
                st.session_state.uploaded_docs.append(doc)
                st.session_state.sec_store.insert(doc)
                st.session_state.uploaded_filenames.add(file_name)

                st.success(f"Document '{file_name}' added to the secondary vector store!")

# Initialize state variables
if "displayed_message_contents" not in st.session_state:
    st.session_state.displayed_message_contents = set()

# Refresh chat history display to avoid duplicates
st.session_state.chat_messages_display = [
    msg for msg in st.session_state.chat_messages
    if msg.content not in st.session_state.displayed_message_contents
    and not st.session_state.displayed_message_contents.add(msg.content)
]

# Display chat messages
for chat in st.session_state.chat_messages_display:
    role = "assistant" if chat.role == MessageRole.ASSISTANT else "user"
    with st.chat_message(role):
        st.markdown(chat.content, unsafe_allow_html=True)

import time
from typing import Dict
def solve_user_input(user_input: str) -> Dict[str, str]:
    current_time = time.time()
    global combined_attached_text
    total_cost = 0
    llm_calls = 1
    query_resolution = dict()
    combined_context = ""
    myres = None
    st.session_state.message_counter += 1
    st.session_state.displayed_message_contents.clear()  # Clear previously displayed contents

    # Add user message to chat history
    user_message = ChatMessage(role=MessageRole.USER, content=user_input)
    st.session_state.chat_messages.append(user_message)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Summarize history periodically
    if st.session_state.message_counter % SUMMARIZE_AFTER == 0:
        with st.spinner("Summarizing history..."):
            summary = summarize_history(st.session_state.chat_messages)
            st.session_state.chat_messages = [
                st.session_state.chat_messages[0],  # Keep first message for context
                ChatMessage(role=MessageRole.ASSISTANT, content=summary)
            ]
            st.success("Chat history summarized!")

    # Analyze user query
    full_prompt = f"{user_input}\n\nAttached Document: {combined_attached_text}" if combined_attached_text else user_input
    with st.spinner("Analyzing user query..."):
        query_type, response, cost = process_user_query(
            gemini_model,
            st.session_state.chat_messages,
            user_input,
            document=combined_attached_text if combined_attached_text else ""
        )
        total_cost += cost

    # Handle responses based on query type
    if query_type == "general":
        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=response)
        st.session_state.chat_messages.append(assistant_message)
        with st.chat_message("assistant"):
            st.markdown(response)
        
        myres = response
        
    
    elif query_type == "direct":
        with st.spinner("Providing direct response..."):
            try:
                response_with_h = get_history_str(st.session_state.chat_messages)
                combined_attached_text = combined_attached_text if combined_attached_text else ""
                chat_result = user_proxy.initiate_chat(recipient=auto_agent, message=response_with_h + "\nAttached document's content:\n" + combined_attached_text)
                total_cost += chat_result.cost['usage_including_cached_inference']['gemini-1.5-flash']['total_tokens']
                assistant_responses = []
                llm_calls += 2

                for message in chat_result.chat_history:
                    if message['name'] == "assistant":
                        assistant_responses.append(message['content'])

                # Combine all assistant responses into one message
                unified_response = "\n\n".join(assistant_responses)
                assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=unified_response)
                st.session_state.chat_messages.append(assistant_message)

                
                with st.chat_message("assistant"):
                    st.markdown(unified_response, unsafe_allow_html=True)
                
                myres = unified_response
            except Exception as e:
                st.error(f"An error occurred: {e}")

    # Handle responses based on query type
    elif query_type == "context":
        with st.spinner("Retrieving information"):
            try:
                # Initial retrieval of context
                retriever.similarity_top_k = 5  # Reset retrieval depth
                context_text = "\n".join([doc.text for doc in retriever.retrieve(response)])
                sec_context_text = "\n".join([doc.text for doc in st.session_state.sec_store.as_retriever().retrieve(response)])
                combined_context = f"Database Context:\n{context_text}\n\nUser Document Context:\n{sec_context_text}"
                # Evaluate sufficiency
                if not evaluate_sufficiency(combined_context, response, total_cost):
                    retriever.similarity_top_k += 5  # Increase retrieval depth
                    additional_context = "\n".join([doc.text for doc in retriever.retrieve(response)])
                    additional_sec_context = "\n".join([doc.text for doc in st.session_state.sec_store.as_retriever().retrieve(response)])
                    combined_context += f"\n\nAdditional Context:\n{additional_context}\n\nAdditional User Document Context:\n{additional_sec_context}"
                    llm_calls +=2

                    # Re-evaluate sufficiency
                    if not evaluate_sufficiency(combined_context, response, total_cost):
                        # Use web search tool if still insufficient
                        llm_calls +=2

                        search_results = web_search_with_logging(response)
                        combined_context += f"\n\nWeb Search Results:\n{search_results}"
                        print(f"Search result: {search_results}")
                        # Final sufficiency check
                        if not evaluate_sufficiency(combined_context, response, total_cost):
                            # Notify the user about insufficiency
                            llm_calls +=2
                            assistant_message = ChatMessage(
                                role=MessageRole.ASSISTANT,
                                content=(
                                    "I couldn't find sufficient context to fully answer your query based on available documents and web search. "
                                    "Please provide a clarified query."
                                ),
                            )
                            st.session_state.chat_messages.append(assistant_message)
                            with st.chat_message("assistant"):
                                st.markdown(assistant_message.content)
                        else:
                            # Process sufficient context
                            formatted_messages = [
                                {"role": "user" if msg.role == MessageRole.USER else "assistant", "content": msg.content}
                                for msg in st.session_state.chat_messages
                            ]
                            formatted_messages.append({"role": "user", "content": f"{response}\nContext:\n{combined_context}"})
                            response_with_h = get_history_str(st.session_state.chat_messages) + f"\nContext:\n{combined_context}"
                            chat_result = user_proxy.initiate_chat(recipient=auto_agent, message=response_with_h)
                            total_cost += chat_result.cost['usage_including_cached_inference']['gemini-1.5-flash']['total_tokens']

                            assistant_responses = [message['content'] for message in chat_result.chat_history if message['name'] == "assistant"]
                            llm_calls +=2
                            # Combine all assistant responses into one message
                            unified_response = "\n\n".join(assistant_responses)
                            assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=unified_response)

                            st.session_state.chat_messages.append(assistant_message)

                            with st.chat_message("assistant"):
                                st.markdown(unified_response, unsafe_allow_html=True)

                            myres = unified_response
                
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Invalid query type detected. Please try again.")
    
    query_resolution['retrieved_context'] = combined_context
    query_resolution['response'] = myres
    query_resolution['llm_calls'] = llm_calls
    query_resolution['total_cost'] = total_cost
    end_time = time.time()
    query_resolution['time_taken'] = end_time - current_time
    return query_resolution

# Chat input
if user_input := st.chat_input("Enter your chat prompt:"):
    qu = solve_user_input(user_input)
    print(qu)