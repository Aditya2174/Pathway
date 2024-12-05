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
import pdfplumber
import pandas as pd
import tiktoken
import traceback
from datetime import datetime

# All LlamaIndex tools needed...LLM, memory, roles, etc
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from utils import (
    process_user_query,
    get_colored_text,
    build_prompt,
    hyde,
    get_num_tokens,
    classify_query,
    is_plot_in_response,
    summarize_history,
    read_file_from_cache_or_parse,
    evaluate_sufficiency
)
from lsa import clustered_rag_lsa
from google_search import web_search_with_logging
from prompts import (
    agent_system_prompt,
    user_proxy_prompt
)
# from guardrail import ChatModerator
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
hf_token = "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"
if not os.environ.get('TAVILY_API_KEY'):
    os.environ['TAVILY_API_KEY'] = tavily_api_key

if not os.environ.get('HF_TOKEN'):
    os.environ['HF_TOKEN'] = hf_token

tavily = TavilyClient(tavily_api_key)

def search_tool(query: Annotated[str, "The search query"]) -> Annotated[str, "The search results"]:
    return tavily.get_search_context(query=query, search_depth="advanced")


if not os.environ.get('AUTOGEN_USE_DOCKER'):
    os.environ['AUTOGEN_USE_DOCKER'] = '0'

google_api_key = "AIzaSyAjcXdnMAYrMXLv0cml6MNQF5udbV-F4xo"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Initialize the WebSearchAPI
# GOOGLE_CSE_ID = "AIzaSyBKTZOFWvwR-GMQvSEKthzOpDU86CB8Zoc"
google_cse_id = "d3a75edbda16e451e"

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

if "document_cache" not in st.session_state:
    st.session_state.document_cache = {}  # Initialize with a default value


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
if 'plot_count' not in st.session_state:
    st.session_state.plot_count = 1

# @st.cache_resource
# def get_moderator():
#     # login()  #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 moi tokennn.
#     return ChatModerator(model_id="meta-llama/Llama-Guard-3-8B", device=device)

# st.session_state.moderator = get_moderator()

executor = LocalCommandLineCodeExecutor(work_dir="coding", timeout=15)
agent_model_name = "gemini-1.5-flash"

auto_agent = ConversableAgent(name="assistant", human_input_mode="NEVER", system_message=agent_system_prompt.format(current_date = datetime.now().strftime("%Y-%m-%d"), plot_count=st.session_state.plot_count),
                                llm_config={"config_list": [{"model": agent_model_name, "temperature": 0.5, "api_key": os.environ.get("GOOGLE_API_KEY"), "api_type": "google"}]},
                                code_execution_config=False)

user_proxy_code = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config={'executor': executor},
                            default_auto_reply=user_proxy_prompt)

user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config=False,
                            default_auto_reply=user_proxy_prompt)

LARGE_FILE_DIRECTORY = '../data/'

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
                    SMALL_FILE_SIZE_THRESHOLD,
                    LARGE_FILE_DIRECTORY
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
        if is_plot_in_response(chat.content):
            image_idx = chat.content.split('plots/image_')[1].split('.png')[0]
            st.image(f'coding/plots/image_{image_idx}.png')

from typing import Dict
def solve_user_query(user_input:str) -> Dict[str, str]:
    global combined_attached_text
    output = dict()
    result = 'safe'    
    combined_context = ""
    llm_calls = 1
    total_token_cost = 0
    assistant_message = None
    stored_response = None
    unified_response = None
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

        with st.status("Analyzing user query...", expanded=False) as status1:
            # contextualized_prompt = contextualize_prompt(user_input)
            document_txt = combined_attached_text if combined_attached_text else ""
            query_type, response, cost = process_user_query(gemini_model, st.session_state.chat_messages, user_input, document=document_txt)
            total_token_cost += cost
            st.write(f"Query Type: {query_type}")
            st.write(f"Reformed query: {response}")
            st.write(f"Token cost: {cost}")
        status1.update(label=f'Query Type: {query_type}', expanded=False, state='complete')

        # Handle responses based on query type
        if "general" in query_type.lower():
            assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=response)
            stored_response = response
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
            st.session_state.chat_messages.append(assistant_message)
            st.session_state.chat_messages_display.append(ChatMessage(role=MessageRole.USER, content=user_input))
            st.session_state.chat_messages_display.append(assistant_message)
            with st.chat_message("assistant"):
                st.markdown(response)
        else:
            green_txt_reform = get_colored_text(response)
            # st.markdown(green_txt_reform, unsafe_allow_html=True)
            display_msg = user_input + "<br>" + green_txt_reform
            st.session_state.chat_messages_display.append(ChatMessage(role=MessageRole.USER, content=display_msg))
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=response))

            with st.status(label="Classifying user query...", expanded=False) as status5:
                query_class = classify_query(gemini_model, user_input, query_type)
                st.write("Query Classification:", query_class)
            status5.update(label=f'Query class: {query_class}')

            if 'direct' in query_type.lower():
                print("Query Type: Direct")
                with st.status("Generating response...", expanded=False) as status2:
                    st.write("Query Type: Direct")
                    try:

                        # TODO: Specify the number of chunks based on query_class
                        sec_store_retrieved = st.session_state.sec_store.as_retriever().retrieve(response)

                        final_prompt = build_prompt(response, chat_history = st.session_state.chat_messages,
                                                    doc_context=[doc.text for doc in sec_store_retrieved])

                        if 'code_execution' in query_class.lower():
                            chat_result = user_proxy_code.initiate_chat(recipient=auto_agent, message=final_prompt)
                        else:
                            chat_result = user_proxy.initiate_chat(recipient=auto_agent, message=final_prompt)
                        assistant_responses = []

                        for message in chat_result.chat_history:
                            if message['name'] == "assistant":
                                llm_calls += 1
                                resp_len = len(message['content'])
                                if 'done' not in message['content'].lower() and resp_len > 5:
                                    assistant_responses.append(message['content'])

                        # Combine all assistant responses into one message
                        unified_response = "\n\n".join(assistant_responses)
                        total_token_cost += chat_result.cost['usage_including_cached_inference'][agent_model_name]['total_tokens']
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

            elif "context" in query_type.lower():
                print("Query Type: Context")
                sufficient = None
                with st.status("Retrieving data...", expanded=False) as status3:
                    try:
                        if 'code_execution' in query_class.lower():
                            retriever.similarity_top_k = 5
                            token_len = get_num_tokens(response)
                            print(f"no of tokens: {token_len}")
                            if(token_len<40):
                                response, cost = hyde(response, gemini_model)
                                total_token_cost += cost
                            print(response)
                            primary_context = retriever.retrieve(response)
                            sec_context = st.session_state.sec_store.as_retriever().retrieve(response)
                            # combined_context = f"Database Context:\n{context_text}\n\nUser Document Context:\n{sec_context_text}"
                            final_prompt = build_prompt(response, chat_history = st.session_state.chat_messages,
                                                        doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context])
                            
                            sufficient = True

                        elif 'summary' in query_class.lower():
                            # Retrieval of large context & then summarizing using LSA
                            retriever.similarity_top_k = 50  # Reset retrieval depth
                            token_len = get_num_tokens(response)
                            print(f"no of tokens: {token_len}")
                            if(token_len<40):
                                response, cost = hyde(response, gemini_model)
                            print(response)

                            primary_store_retrieved = retriever.retrieve(response)
                            st.write(f"Retrieved Database context size: {sum([get_num_tokens(doc.text) for doc in primary_store_retrieved])}")
                            context_summaries = clustered_rag_lsa([doc.text for doc in primary_store_retrieved], num_clusters=20, sentences_count=3)

                            sec_store_retrieved = st.session_state.sec_store.as_retriever().retrieve(response)
                            st.write(f"Retrieved Document context size: {sum([get_num_tokens(doc.text) for doc in sec_store_retrieved])}")
                            document_context = [doc.text for doc in sec_store_retrieved]

                            sec_context_summaries = clustered_rag_lsa(document_context, num_clusters=int(len(document_context)*0.5), sentences_count=5)

                            # combined_context = f"Database Context:\n{context_text}\n\nUser Document Context:\n{sec_context_text}"
                            # st.write(f"LSA token count: {get_num_tokens(combined_context)}")
                            st.write(combined_context)
                            sufficient, cost = evaluate_sufficiency(gemini_model, combined_context, response, total_token_cost)

                            final_prompt = build_prompt(response, chat_history=st.session_state.chat_messages,
                                                        doc_context=sec_context_summaries, retrieved_context=context_summaries)
                        else:
                            # Initial retrieval of context
                            retriever.similarity_top_k = 5  # Reset retrieval depth
                            token_len = get_num_tokens(response)
                            print(f"no of tokens: {token_len}")
                            if(token_len<40):
                                response, cost = hyde(response, gemini_model)
                                total_token_cost += cost
                            print(response)

                            primary_context = retriever.retrieve(response)
                            # context_text = "\n".join([doc.text for doc in retriever.retrieve(response)])
                            sec_context = st.session_state.sec_store.as_retriever().retrieve(response)
                            # sec_context_text = "\n".join([doc.text for doc in st.session_state.sec_store.as_retriever().retrieve(response)])

                            # combined_context = f"Database Context:\n{context_text}\n\nUser Document Context:\n{sec_context_text}"
                            final_prompt = build_prompt(response, chat_history=st.session_state.chat_messages,
                                                        doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context])
                            llm_calls += 1
                            sufficient, cost = evaluate_sufficiency(gemini_model, combined_context, response, total_token_cost)
                            
                            if not sufficient:
                                retriever.similarity_top_k += 5  # Increase retrieval depth
                                primary_context = retriever.retrieve(response)
                                # additional_context = "\n".join([doc.text for doc in retriever.retrieve(response)])
                                sec_context = st.session_state.sec_store.as_retriever().retrieve(response)
                                # additional_sec_context = "\n".join([doc.text for doc in st.session_state.sec_store.as_retriever().retrieve(response)])
                                # combined_context = f"\n\nAdditional Context:\n{additional_context}\n\nAdditional User Document Context:\n{additional_sec_context}"
                                combined_context = build_prompt(response, doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context])
                                final_prompt = build_prompt(response, chat_history=st.session_state.chat_messages,
                                                        doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context])
                                llm_calls += 1 # add an llm call as not sufficient
                        
                            tup = sufficient or evaluate_sufficiency(gemini_model, combined_context, response, total_token_cost)
                            if not sufficient:
                                sufficient = tup[0]
                                cost = tup[1]
                                total_token_cost += cost
                            
                            if not sufficient:
                                # Use web search tool if still insufficient
                                # search_results = web_search_with_logging(response)
                                search_results = search_tool(response)
                                # combined_context += f"\n\nWeb Search Results:\n{search_results}"
                                combined_context = build_prompt(response, doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context], search_results=search_results)
                                print(f"Search result: {search_results}")
                                final_prompt = build_prompt(response, chat_history=st.session_state.chat_messages,
                                                        doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context], search_results=search_results)
                                llm_calls += 1
                            
                            tup = sufficient or evaluate_sufficiency(gemini_model, combined_context, response, total_token_cost)
                            if not sufficient:
                                sufficient = tup[0]
                                cost = tup[1]
                                total_token_cost += cost
                        st.write(combined_context)
                    except Exception as e:
                        st.error(f"An error occurred: {e}")
                        traceback.print_exc()
                status3.update(label="Retrieval Complete!" if sufficient else "Not enough context 🙁", expanded=False)

                if not sufficient:
                    # Notify the user about insufficiency
                    res = """I couldn't find sufficient context to fully answer your query based on available documents and web search. Please provide a clarified query."""
                    assistant_message = ChatMessage(role=MessageRole.ASSISTANT,content=(res))
                    
                    stored_response = res
                    st.session_state.chat_messages.append(assistant_message)
                    with st.chat_message("assistant"):
                        st.markdown(assistant_message.content)
                else:
                    # formatted_messages = [
                    #     {"role": "user" if msg.role == MessageRole.USER else "assistant", "content": msg.content}
                    #     for msg in st.session_state.chat_messages
                    # ]
                    # formatted_messages.append({"role": "user", "content": f"{response}\nContext:\n{combined_context}"})
                    # response_with_h = get_history_str(st.session_state.chat_messages) + f"\nContext:\n{combined_context}"'

                    with st.status("Generating response...", expanded=False) as status2:
                        if 'code_execution' in query_class.lower():
                            chat_result = user_proxy_code.initiate_chat(recipient=auto_agent, message=final_prompt)
                        else:
                            chat_result = user_proxy.initiate_chat(recipient=auto_agent, message=final_prompt)
                        assistant_responses = [message['content'] for message in chat_result.chat_history if (message['name'] == "assistant") and ((message['content'] != "done") and (len(message['content']) > 5))]
                        st.write(chat_result)
                        for message in chat_result.chat_history:
                            llm_calls += (message['name'] == 'assistant')
                            
                        total_token_cost += chat_result.cost['usage_including_cached_inference'][agent_model_name]['total_tokens']
                        # Combine all assistant responses into one message
                        unified_response = "\n\n".join(assistant_responses)
                        stored_response = unified_response
                        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=unified_response)
                        st.session_state.chat_messages.append(assistant_message)

                    with st.chat_message("assistant"):
                        st.markdown(unified_response, unsafe_allow_html=True)
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

    if is_plot_in_response(output['response']):
        st.image(f'coding/plots/image_{st.session_state.plot_count}.png')
        st.session_state.plot_count += 1