"""
Main Frontend file
Please run it using the following command:
streamlit run src/ash_frontend_updated.py
"""
# Streamlit for UI
import streamlit as st
from typing import Dict
st.set_page_config(layout="wide", initial_sidebar_state="auto")

st.markdown("""
    <style>
        .stExpander  {
            max-height: 300px;  /* Define max height for status containers */
            overflow-y: auto;  /* Enable scrolling if content overflows */
        }
    </style>
""", unsafe_allow_html=True)

# For file-related purposes
import os
import shutil
import diskcache as dc
import atexit
import tiktoken
import traceback
from datetime import datetime

# All LlamaIndex tools needed...LLM, memory, roles, etc
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.llms import ChatMessage, MessageRole
from utils import (
    process_user_query,
    build_prompt,
    hyde,
    get_num_tokens,
    classify_query,
    is_plot_in_response,
    summarize_history,
    read_file_from_cache_or_parse,
    evaluate_sufficiency,
    search_tool
)
from analyse_legal import analyze_document_with_context
from lsa import clustered_rag_lsa
from prompts import (
    agent_system_prompt,
    user_proxy_prompt,
    unsafe_query_resp
)
from guardrail import ChatModerator
from huggingface_hub import login
from llama_index.core.indices import VectorStoreIndex
from llama_index.core import Document
from llama_index.embeddings.google import GeminiEmbedding
import torch

# Autogen agents
from autogen import ConversableAgent, UserProxyAgent
from autogen.coding import LocalCommandLineCodeExecutor
from sentence_transformers import SentenceTransformer

# Summarizer of history
from transformers import pipeline

from tavily import TavilyClient

# Tavily search tool setup
tavily_api_key = "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"

hf_token = "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"
if not os.environ.get('TAVILY_API_KEY'):
    os.environ['TAVILY_API_KEY'] = tavily_api_key

if not os.environ.get('HF_TOKEN'):
    os.environ['HF_TOKEN'] = hf_token

tavily_client = TavilyClient(tavily_api_key)

if not os.environ.get('AUTOGEN_USE_DOCKER'):
    os.environ['AUTOGEN_USE_DOCKER'] = '0'

google_api_key = "AIzaSyBmj80JwqQpnpgRTJmMc1LEVdJveZu4ApU"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

# Constants/Configurations
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Number of messages after which chat history is summarized
SUMMARIZE_AFTER = 10
SMALL_FILE_SIZE_THRESHOLD = 0.75 * 1024 * 1024  # 0.75 MB

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8756
LSA_CHUNK_RETRIEVAL = 50
LSA_SENTENCE_COUNT = 10
RETRIEVAL_DEPTH = 5

# Cache directory setup using diskcache
cache_dir = './document_cache'
document_cache = dc.Cache(cache_dir)
analysis_dir = './analysis_cache'
analysis_cache = dc.Cache(analysis_dir)

# Register cleanup function to delete cache on termination
def cleanup_cache():
    if os.path.exists(cache_dir):
        shutil.rmtree(cache_dir)
        print(f"Cache directory '{cache_dir}' cleaned up.")
    if os.path.exists(analysis_dir):
        shutil.rmtree(analysis_dir)
        print(f"Cache directory '{analysis_dir} cleaned up.")

atexit.register(cleanup_cache)

os.makedirs("coding", exist_ok=True)  # Create a working directory for code executor

gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)  # Initialize a Gemini-1.5-Flash model with LlamaIndex

retriever = PathwayRetriever(host=PATHWAY_HOST, port=PATHWAY_PORT, similarity_top_k=RETRIEVAL_DEPTH) # Initialize a PathwayRetriever

# Initialize session variables
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []
if 'chat_messages_display' not in st.session_state:
    st.session_state.chat_messages_display = []
if 'summarized_history' not in st.session_state:
    st.session_state.summarized_history = ""
if 'message_counter' not in st.session_state:
    st.session_state.message_counter = 0
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
if 'query_classes' not in st.session_state:
    st.session_state.query_classes = []
if 'analyse_doc_mode' not in st.session_state:
    st.session_state.analyse_doc_mode = False

# Cache resources so that they are not reloaded on every refresh
@st.cache_resource
def get_moderator():
    # login()  #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 moi tokennn.
    return ChatModerator(model_id="meta-llama/Llama-Guard-3-8B", device=device)

@st.cache_resource
def get_lsa_embedder():
    embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    return embedder

@st.cache_resource
def get_chat_summarizer():
    summarizer = pipeline("summarization", device=device, model="facebook/bart-large-cnn")
    return summarizer

moderator = get_moderator()
lsa_embedder = get_lsa_embedder()
chat_summarizer = get_chat_summarizer()

print("All resources loaded successfully!")

executor = LocalCommandLineCodeExecutor(work_dir="coding", timeout=15) # Code executor for the user proxy agent
agent_model_name = "gemini-1.5-flash" # Model name for the generator agent

auto_agent = ConversableAgent(name="assistant", human_input_mode="NEVER", system_message=agent_system_prompt.format(current_date = datetime.now().strftime("%Y-%m-%d"), plot_count=st.session_state.plot_count),
                                llm_config={"config_list": [{"model": agent_model_name, "temperature": 0.5, "api_key": os.environ.get("GOOGLE_API_KEY"), "api_type": "google"}]},
                                code_execution_config=False)

user_proxy_code = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config={'executor': executor},
                            default_auto_reply=user_proxy_prompt) # User proxy agent for code execution

user_proxy = UserProxyAgent(name="user_proxy", human_input_mode="NEVER", max_consecutive_auto_reply=1, code_execution_config=False,
                            default_auto_reply=user_proxy_prompt) # User proxy agent for normal queries

# Directory for storing large files
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

    st.toggle(label = 'Risk analysis mode',
              key = 'analyse_doc_mode',
              disabled = (not uploaded_files)
            )

    if uploaded_files:

        for uploaded_file in uploaded_files:
            file_name = uploaded_file.name

            if (file_name in st.session_state.uploaded_filenames) and (not st.session_state.analyse_doc_mode):
                st.warning(f"'{file_name}' has already been added to the vector store.")
                combined_attached_text += document_cache[file_name] + "\n"
            else:
                attached_text = read_file_from_cache_or_parse(
                    uploaded_file,
                    document_cache,
                    analysis_cache,
                    SMALL_FILE_SIZE_THRESHOLD,
                    LARGE_FILE_DIRECTORY,
                    risk_analysis_mode = st.session_state.analyse_doc_mode
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
idx = 0
for chat in st.session_state.chat_messages_display:
    role = "assistant" if chat.role == MessageRole.ASSISTANT else "user"
    with st.chat_message(role):
        st.write(chat.content, unsafe_allow_html=True)
        if (role == 'assistant') and is_plot_in_response(chat.content) and ('code_execution' in st.session_state.query_classes[idx]):
            # Display plots generated by the code executor
            image_idx = chat.content.split('plots/image_')[1].split('.png')[0]
            st.image(f'coding/plots/image_{image_idx}.png')
        if (role == 'assistant'):
            idx += 1

def rag_pipeline(user_input:str) -> Dict[str, str]:
    """
    Function to process user query and generate response
    """
    global combined_attached_text
    output = dict()
    combined_context = ""
    llm_calls = 1
    total_token_cost = 0
    assistant_message = None
    stored_response = None
    unified_response = None
    final_prompt = None
    st.session_state.message_counter += 1
    st.session_state.displayed_message_contents.clear()  # Clear previously displayed contents
    sec_context = None
    primary_context = None

    with st.chat_message("user"):
        st.write(user_input)

        # Summarize history periodically
    if st.session_state.message_counter % SUMMARIZE_AFTER == 0:
        with st.status("Summarizing history...", expanded=False) as status0:
            st.session_state.summarized_history = summarize_history(chat_summarizer, st.session_state.chat_messages)
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
            st.session_state.chat_messages = [
                st.session_state.chat_messages[0],
                ChatMessage(role=MessageRole.ASSISTANT, content=st.session_state.summarized_history)
            ]
            st.write("Summarised History:\n"+st.session_state.summarized_history)
        status0.update(label="Summarized successfully", expanded=False)

    with st.status("Analyzing user query...", expanded=False) as status1:
        document_txt = combined_attached_text if combined_attached_text else ""

        # Reformulation and query processing
        query_type, response, cost = process_user_query(gemini_model, st.session_state.chat_messages, user_input, document=document_txt)
        total_token_cost += cost

        st.write(f"Query Type: {query_type}")
        st.write(f"Reformed query: {response}")
        st.write(f"Token cost: {cost}")

    status1.update(label=f'Query Type: {query_type}', expanded=False, state='complete')

    # Handle responses based on query type
    if "general" in query_type.lower():
        # General chi-chat response
        assistant_message = ChatMessage(role=MessageRole.ASSISTANT, content=response)
        stored_response = response

        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
        st.session_state.chat_messages.append(assistant_message)

        with st.chat_message("assistant"):
            st.write(response)

        st.session_state.query_classes.append(None)
    else:
        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=response))

        with st.status(label="Classifying user query...", expanded=False) as status5:
            # Classify the query into different categories
            query_class = classify_query(gemini_model, user_input, query_type)

            st.session_state.query_classes.append(query_class)
            st.write("Query Classification:", query_class)
        status5.update(label=f'Query class: {query_class}')

        if 'direct' in query_type.lower():
            # Respond directly from the user document
            print("Query Type: Direct")
            with st.status("Generating response...", expanded=False) as status2:
                st.write("Query Type: Direct")
                try:
                    if 'summary' in query_class.lower():
                        # Retrieve from secondary vector store
                        sec_store_retrieved = st.session_state.sec_store.as_retriever(similarity_top_k = LSA_CHUNK_RETRIEVAL).retrieve(response)

                        st.write(f"Retrieved Document context size: {sum([get_num_tokens(doc.text) for doc in sec_store_retrieved])}")

                        document_context = [doc.text for doc in sec_store_retrieved]

                        sec_context = clustered_rag_lsa(lsa_embedder, document_context, num_clusters=int(len(document_context)*0.5), sentences_count=LSA_SENTENCE_COUNT)

                    else:
                        sec_context = st.session_state.sec_store.as_retriever().retrieve(response)

                    try:
                        final_prompt = build_prompt(response, chat_history = st.session_state.chat_messages,
                                                doc_context=[doc.text for doc in sec_context])
                    except:
                        final_prompt = build_prompt(response, chat_history = st.session_state.chat_messages,
                                                doc_context=[doc for doc in sec_context])

                    if 'code_execution' in query_class.lower():
                        # Pass it to the user proxy agent with code execution enabled
                        chat_result = user_proxy_code.initiate_chat(recipient=auto_agent, message=final_prompt)
                    else:
                        # Pass it to the user proxy agent
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

                    st.write(unified_response, unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

            with st.chat_message("assistant"):
                st.write(unified_response, unsafe_allow_html=True)

            st.session_state.chat_messages.append(assistant_message)
            status2.update(label="Generation Complete!", expanded=False)

        elif "context" in query_type.lower():
            # Respond after retrieving context
            print("Query Type: Context")
            sufficient = None
            with st.status("Retrieving data...", expanded=False) as status3:
                try:
                    if 'code_execution' in query_class.lower():
                        retriever.similarity_top_k = RETRIEVAL_DEPTH
                        token_len = get_num_tokens(response)
                        print(f"no of tokens: {token_len}")
                        print(response)
                        primary_context = retriever.retrieve(response)
                        sec_context = st.session_state.sec_store.as_retriever().retrieve(response)
                        final_prompt = build_prompt(response, chat_history = st.session_state.chat_messages,
                                                    doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context])
                        
                        combined_context = build_prompt(response, doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context])
                        st.text(combined_context)
                        sufficient = True

                    elif 'summary' in query_class.lower():
                        # Retrieval of large context & then summarizing using LSA
                        retriever.similarity_top_k = LSA_CHUNK_RETRIEVAL  # Reset retrieval depth
                        token_len = get_num_tokens(response)

                        print(f"no of tokens: {token_len}")
                        print(response)

                        primary_store_retrieved = retriever.retrieve(response)

                        st.write(f"Retrieved Database context size: {sum([get_num_tokens(doc.text) for doc in primary_store_retrieved])}")

                        context_summaries = clustered_rag_lsa(lsa_embedder, [doc.text for doc in primary_store_retrieved], num_clusters=20, sentences_count=LSA_SENTENCE_COUNT)

                        sec_store_retrieved = st.session_state.sec_store.as_retriever().retrieve(response)

                        st.write(f"Retrieved Document context size: {sum([get_num_tokens(doc.text) for doc in sec_store_retrieved])}")

                        document_context = [doc.text for doc in sec_store_retrieved]

                        sec_context_summaries = clustered_rag_lsa(lsa_embedder, document_context, num_clusters=int(len(document_context)*0.5), sentences_count=LSA_SENTENCE_COUNT)

                        combined_context = build_prompt(response, doc_context=sec_context_summaries, retrieved_context=context_summaries)

                        st.write(f"LSA token count: {get_num_tokens(combined_context)}")
                        st.text(combined_context)

                        # Evaluate the sufficiency of the current context
                        sufficient, cost = evaluate_sufficiency(gemini_model, combined_context, response, total_token_cost)

                        final_prompt = build_prompt(response, chat_history=st.session_state.chat_messages,
                                                    doc_context=sec_context_summaries, retrieved_context=context_summaries)
                        sec_context = sec_context_summaries
                        primary_context = context_summaries
                    else:
                        # Initial retrieval of context
                        retriever.similarity_top_k = RETRIEVAL_DEPTH  # Reset retrieval depth
                        token_len = get_num_tokens(response)

                        print(f"no of tokens: {token_len}")
                        print(response)

                        primary_context = retriever.retrieve(response)
                        sec_context = st.session_state.sec_store.as_retriever().retrieve(response)

                        combined_context = build_prompt(response, doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context])

                        llm_calls += 1

                        # Evaluate the sufficiency of the current context
                        sufficient, cost = evaluate_sufficiency(gemini_model, combined_context, response, total_token_cost)
                        
                        if not sufficient:
                            # If not sufficient, increase retrieval depth
                            retriever.similarity_top_k += RETRIEVAL_DEPTH  # Increase retrieval depth

                            primary_context = retriever.retrieve(response)
                            sec_context = st.session_state.sec_store.as_retriever().retrieve(response)

                            combined_context = build_prompt(response, doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context])
                            llm_calls += 1 # add an llm call as not sufficient

                        # Again evaluate the sufficiency of the current context
                        tup = sufficient or evaluate_sufficiency(gemini_model, combined_context, response, total_token_cost)
                        if not sufficient:
                            sufficient = tup[0]
                            cost = tup[1]
                            total_token_cost += cost
                        
                    if not sufficient:
                        # Use web search tool if still insufficient
                        search_results = search_tool(response, tavily_client)

                        try:
                            # Build prompt with search results and a combination of retrieved and document context
                            combined_context = build_prompt(response, doc_context=[doc.text for doc in sec_context[:RETRIEVAL_DEPTH]], retrieved_context=[doc.text for doc in primary_context[:RETRIEVAL_DEPTH]], search_results=search_results)
                            final_prompt = build_prompt(response, chat_history=st.session_state.chat_messages,
                                                doc_context=[doc.text for doc in sec_context[:RETRIEVAL_DEPTH]], retrieved_context=[doc.text for doc in primary_context], search_results=search_results[:RETRIEVAL_DEPTH])
                        except:
                            combined_context = build_prompt(response, doc_context=[doc for doc in sec_context[:RETRIEVAL_DEPTH]], retrieved_context=[doc for doc in primary_context[:RETRIEVAL_DEPTH]], search_results=search_results)
                            final_prompt = build_prompt(response, chat_history=st.session_state.chat_messages,
                                                doc_context=[doc.text for doc in sec_context[:RETRIEVAL_DEPTH]], retrieved_context=[doc.text for doc in primary_context[:RETRIEVAL_DEPTH]], search_results=search_results)

                        llm_calls += 1
                    else:
                        try:
                            final_prompt = build_prompt(response, chat_history=st.session_state.chat_messages,
                                                    doc_context=[doc.text for doc in sec_context], retrieved_context=[doc.text for doc in primary_context])
                        except:
                            final_prompt = build_prompt(response, chat_history=st.session_state.chat_messages,
                                                    doc_context=[doc for doc in sec_context], retrieved_context=[doc for doc in primary_context])
                    
                    # Check the sufficiency again
                    tup = sufficient or evaluate_sufficiency(gemini_model, combined_context, response, total_token_cost)
                    if not sufficient:
                        sufficient = tup[0]
                        cost = tup[1]
                        total_token_cost += cost
 
                    st.text(combined_context)
                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    traceback.print_exc()
            status3.update(label="Retrieval Complete!" if sufficient else "Not enough context 🙁", expanded=False)

            if not sufficient:
                # Notify the user about insufficiency
                res = """Sorry, I couldn't find sufficient context to fully answer your query based on available documents and web search. Please provide a clarified query."""
                assistant_message = ChatMessage(role=MessageRole.ASSISTANT,content=res)
                
                stored_response = res
                st.session_state.chat_messages.append(assistant_message)
                with st.chat_message("assistant"):
                    st.write(assistant_message.content)
            else:
                with st.status("Generating response...", expanded=False) as status2:
                    print(final_prompt)

                    if 'code_execution' in query_class.lower():
                        # Pass it to the user proxy agent with code execution enabled
                        chat_result = user_proxy_code.initiate_chat(recipient=auto_agent, message=final_prompt)
                    else:
                        # Pass it to the user proxy agent
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
                    st.write(unified_response, unsafe_allow_html=True)
                status2.update(label="Generation Complete!", expanded=False)

        else:
            print("Invalid response type detected:", query_type)
            st.error("Invalid response type detected. Please try again.")
    
    output['retrieved_contexts'] = combined_context
    output['response'] = stored_response
    output['llm_calls'] = llm_calls
    output['total_token_cost'] = total_token_cost
    return output

# Chat input
if user_input := st.chat_input("Enter your chat prompt:"):
    # Guardrail to prevent unsafe queries
    result = moderator.moderate_chat([{"role": "user", "content": user_input}])

    print(f"Guardrail result: {result}")
    result = result.split('\n')[0]

    if not 'unsafe' in result.lower():
        # Safe query
        if st.session_state.analyse_doc_mode:
            with st.chat_message("user"):
                st.write(user_input)
            
            with st.spinner("Getting the response..."):
                response = analyze_document_with_context(gemini_model, combined_attached_text, user_input)
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))

            with st.chat_message("assistant"):
                st.write(response)

        else:
            # Pass the user query to RAG pipeline
            output = rag_pipeline(user_input)

            if is_plot_in_response(output['response']) and ('code_execution' in st.session_state.query_classes[-1]):
                st.image(f'coding/plots/image_{st.session_state.plot_count}.png')
                st.session_state.plot_count += 1
    else:
        # Unsafe query
        print("Unsafe query detected:")
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            st.write(unsafe_query_resp)
        
        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=unsafe_query_resp))