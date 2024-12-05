# Streamlit for UI
import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="auto")

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
import tiktoken
from datetime import datetime

# All LlamaIndex tools needed...LLM, memory, roles, etc
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from utils import (
    process_user_query,
    get_colored_text,
    get_history_str,
    hyde,
    get_num_tokens
)
from prompts import (
    agent_system_prompt,
    user_proxy_prompt
)

from google_search import web_search_with_logging
from guardrail import ChatModerator
from huggingface_hub import login
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
from tavily import TavilyClient

# Tavily search tool setup
tavily_api_key = "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"
if not os.environ.get('TAVILY_API_KEY'):
    os.environ['TAVILY_API_KEY'] = tavily_api_key

tavily = TavilyClient(tavily_api_key)

def search_tool(query: Annotated[str, "The search query"]) -> Annotated[str, "The search results"]:
    return tavily.get_search_context(query=query, search_depth="advanced")
from typing import Tuple
def evaluate_sufficiency(context_text: str, query: str, total_cost :int) -> Tuple[bool, int]:
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
        if 'no' in result.message.content.lower():
            return False, cost
        return True, cost

if not os.environ.get('AUTOGEN_USE_DOCKER'):
    os.environ['AUTOGEN_USE_DOCKER'] = '0'

google_api_key = "AIzaSyAjcXdnMAYrMXLv0cml6MNQF5udbV-F4xo"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

tool_usage_log = {} # Dictionary to track tool usage

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
if 'tiktoken_tokenizer' not in st.session_state:
    st.session_state.tiktoken_tokenizer = tiktoken.encoding_for_model('gpt-4')
# if 'moderator' not in st.session_state:
#     login()  #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 moi tokennn.
#     st.session_state.moderator = ChatModerator(model_id="meta-llama/Llama-Guard-3-8B")

executor = LocalCommandLineCodeExecutor(work_dir="coding", timeout=15)
agent_model_name = "gemini-1.5-flash"

auto_agent = ConversableAgent(name="assistant", human_input_mode="NEVER", system_message=agent_system_prompt.format(current_date = datetime.now().strftime("%Y-%m-%d")),
                                llm_config={"config_list": [{"model": agent_model_name, "temperature": 0.5, "api_key": os.environ.get("GOOGLE_API_KEY"), "api_type": "google"}]},
                                code_execution_config=False)

user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config={'executor': executor},
                            default_auto_reply=user_proxy_prompt)

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
    combined_attached_text = ""

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

from typing import Dict
def solve_user_query(user_input:str) -> Dict[str, str]:
    global combined_attached_text
    output = dict()
    result = 'safe'    
    combined_context = ""
    llm_calls = 1
    total_token_cost = 0
    if result == 'safe':
        st.session_state.message_counter += 1
        st.session_state.displayed_message_contents.clear()  # Clear previously displayed contents

        with st.chat_message("user"):
            st.markdown(user_input)

            # Summarize history periodically
        if st.session_state.message_counter % SUMMARIZE_AFTER == 0:
            with st.status("Summarizing history...", expanded=False) as status0:
                st.session_state.summarized_history = summarize_history(st.session_state.chat_messages)
                st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
                st.session_state.chat_messages = [
                    st.session_state.chat_messages[0],
                    ChatMessage(role=MessageRole.ASSISTANT, content=st.session_state.summarized_history)
                ]
                st.write("Summarised History:\n"+st.session_state.summarized_history)
            status0.update(label="Summarized successfully", expanded=False)

        full_prompt = f"{user_input}\n\nAttached Document: {combined_attached_text}" if combined_attached_text else user_input
        with st.status("Analyzing user query...", expanded=False) as status1:
            # contextualized_prompt = contextualize_prompt(user_input)
            document_txt = combined_attached_text if combined_attached_text else ""
            query_type, response, cost = process_user_query(gemini_model, st.session_state.chat_messages, user_input, document=document_txt)
            total_token_cost += cost
            st.write(f"Query Type: {query_type}")
            st.write(f"Reformed query: {response}")
        status1.update(label=f'Query Type: {query_type}', expanded=False, state='complete')

        # Handle responses based on query type
        if query_type == "general":
            assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=response)
            stored_response = response
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
            st.session_state.chat_messages.append(assistant_message)
            st.session_state.chat_messages_display.append(ChatMessage(role=MessageRole.USER, content=user_input))
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            green_txt_reform = get_colored_text(response)
            # st.markdown(green_txt_reform, unsafe_allow_html=True)
            display_msg = user_input + "<br>" + green_txt_reform
            st.session_state.chat_messages_display.append(ChatMessage(role=MessageRole.USER, content=display_msg))
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=response))
            if query_type == 'direct':
                print("Query Type: Direct")
                if combined_attached_text:
                    # Modify the last message in the session state to include document context
                    last_message = st.session_state.chat_messages[-1]
                    last_message.content = f"{last_message.content}\n\nAttached Document Context:\n{combined_attached_text}"

                formatted_messages = [
                    {"role": "user" if msg.role == MessageRole.USER else "assistant", "content": msg.content}
                    for msg in st.session_state.chat_messages
                ]
                response_with_h = get_history_str(st.session_state.chat_messages)
                combined_attached_text = combined_attached_text if combined_attached_text else ""
                with st.status("Generating response...", expanded=False) as status2:
                    st.write("Query Type: Direct")
                    try:
                        
                        chat_result = user_proxy.initiate_chat(recipient=auto_agent, message=response_with_h + "\nAttached document's content:\n" + combined_attached_text)
                        assistant_responses = []

                        for message in chat_result.chat_history:
                            if message['name'] == "assistant":
                                llm_calls += 1
                                assistant_responses.append(message['content'])

                        # Combine all assistant responses into one message
                        unified_response = "\n\n".join(assistant_responses)
                        total_cost += chat_result.cost['usage_including_cached_inference'][agent_model_name]['total_tokens']
                        stored_response = unified_response
                        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=unified_response)
                        st.markdown(unified_response, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

                    # print(f"Chat Result:\n{chat_result['chat_history'][-1]['content']}")
                with st.chat_message("assistant"):
                    st.markdown(unified_response, unsafe_allow_html=True)
                st.session_state.chat_messages.append(assistant_message)
                st.session_state.chat_messages_display.append(assistant_message)
                status2.update(label="Generation Complete!", expanded=False)

            elif query_type == "context":
                print("Query Type: Context")
                sufficient = None
                with st.status("Retrieving data...", expanded=False) as status3:
                    try:
                        # Initial retrieval of context
                        retriever.similarity_top_k = 5  # Reset retrieval depth
                        token_len = get_num_tokens(response)
                        print(f"no of tokens: {token_len}")
                        if(token_len<40):
                            response, cost = hyde(response, gemini_model)
                            total_token_cost += cost
                        print(response)
                        context_text = "\n".join([doc.text for doc in retriever.retrieve(response)])
                        sec_context_text = "\n".join([doc.text for doc in st.session_state.sec_store.as_retriever().retrieve(response)])
                        combined_context = f"Database Context:\n{context_text}\n\nUser Document Context:\n{sec_context_text}"
                        llm_calls += 1
                        sufficient, cost = evaluate_sufficiency(combined_context, response)
                        
                        if not sufficient:
                            retriever.similarity_top_k += 5  # Increase retrieval depth
                            additional_context = "\n".join([doc.text for doc in retriever.retrieve(response)])
                            additional_sec_context = "\n".join([doc.text for doc in st.session_state.sec_store.as_retriever().retrieve(response)])
                            combined_context += f"\n\nAdditional Context:\n{additional_context}\n\nAdditional User Document Context:\n{additional_sec_context}"
                            llm_calls += 1 # add an llm call as not sufficient
                        
                        
                        tup = sufficient or evaluate_sufficiency(combined_context, response)
                        if not sufficient:
                            sufficient = tup[0]
                            cost = tup[1]
                            total_token_cost += cost
                            
                        if not sufficient:
                            # Use web search tool if still insufficient
                            # search_results = web_search_with_logging(response)
                            search_results = search_tool(response)
                            combined_context += f"\n\nWeb Search Results:\n{search_results}"
                            print(f"Search result: {search_results}")
                            llm_calls += 1
                            
                        tup = sufficient or evaluate_sufficiency(combined_context, response)
                        if not sufficient:
                            sufficient = tup[0]
                            cost = tup[1]
                            total_token_cost += cost
                        st.write(combined_context)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                status3.update(label="Retrieval Complete!" if sufficient else "Not enough contex 🙁", expanded=False)

                if not sufficient:
                    # Notify the user about insufficiency
                    res = """I couldn't find sufficient context to fully answer your query based on available documents and web search. "
                            Please provide a clarified query."""
                    assistant_message = ChatMessage(
                        role=MessageRole.ASSISTANT,
                        content=(
                            res
                        ),
                    )
                    
                    stored_response = res
                    st.session_state.chat_messages.append(assistant_message)
                    with st.chat_message("assistant"):
                        st.markdown(assistant_message.content)
                else:
                    formatted_messages = [
                        {"role": "user" if msg.role == MessageRole.USER else "assistant", "content": msg.content}
                        for msg in st.session_state.chat_messages
                    ]
                    formatted_messages.append({"role": "user", "content": f"{response}\nContext:\n{combined_context}"})
                    response_with_h = get_history_str(st.session_state.chat_messages) + f"\nContext:\n{combined_context}"

                    with st.status("Generating response...", expanded=False) as status2:
                        chat_result = user_proxy.initiate_chat(recipient=auto_agent, message=response_with_h)
                        assistant_responses = [message['content'] for message in chat_result.chat_history if message['name'] == "assistant"]
                        st.write(chat_result)
                        for message in chat_result.chat_history:
                            llm_calls += (message['name'] == 'assistant')
                            
                        total_cost += chat_result.cost['usage_including_cached_inference'][agent_model_name]['total_tokens']
                        # Combine all assistant responses into one message
                        unified_response = "\n\n".join(assistant_responses)
                        stored_response = unified_response
                        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=unified_response)
                        st.session_state.chat_messages.append(assistant_message)

                    with st.chat_message("assistant"):
                        st.markdown(unified_response, unsafe_allow_html=True)
                    status2.update(label="Generation Complete!", expanded=False)
            elif query_type == 'code_execution':
                print("Query Type: Code Execution")
                if combined_attached_text:
                    # Modify the last message in the session state to include document context
                    last_message = st.session_state.chat_messages[-1]
                    last_message.content = f"{last_message.content}\n\nAttached Document Context:\n{combined_attached_text}"

                formatted_messages = [
                    {"role": "user" if msg.role == MessageRole.USER else "assistant", "content": msg.content}
                    for msg in st.session_state.chat_messages
                ]
                response_with_h = get_history_str(st.session_state.chat_messages)
                combined_attached_text = combined_attached_text if combined_attached_text else ""
                with st.status("Generating response...", expanded=False) as status2:
                    st.write("Query Type: Direct")
                    try:
                        chat_result = user_proxy.initiate_chat(recipient=auto_agent, message=response_with_h + "\nAttached document's content:\n" + combined_attached_text)
                        assistant_responses = []

                        for message in chat_result.chat_history:
                            if message['name'] == "assistant":
                                assistant_responses.append(message['content'])
                                llm_calls+= 1
                        # Combine all assistant responses into one message
                        unified_response = "\n\n".join(assistant_responses)
                        total_cost += chat_result.cost['usage_including_cached_inference'][agent_model_name]['total_tokens']
                        stored_response = unified_response
                        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=unified_response)
                        st.markdown(unified_response, unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")

                
                with st.chat_message("assistant"):
                    st.markdown(unified_response, unsafe_allow_html=True)
                st.session_state.chat_messages.append(assistant_message)
                st.session_state.chat_messages_display.append(assistant_message)
                status2.update(label="Generation Complete!", expanded=False)

            else:
                print("Invalid response type detected:", query_type)
                st.error("Invalid response type detected. Please try again.")
    else:
        print("Unsafe query detected:")
        st.error("Unsafe query detected. Please try again.")
    print(st.session_state.chat_messages)
    
    output['retrieved_contexts'] = combined_context
    output['response'] = stored_response
    output['llm_calls'] = llm_calls
    output['total_token_cost'] = total_token_cost
    return output
# Chat input
if user_input := st.chat_input("Enter your chat prompt:"):
    # result = st.session_state.moderator.moderate_chat([{"role": "user", "content": user_input}])
    # print(result)
    # result = result.split('\n')[0]
    
    output = solve_user_query(user_input)
    print(output)
