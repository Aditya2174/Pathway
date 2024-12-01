# Streamlit for UI
import streamlit as st

# For file-related purposes
import os
import shutil
import diskcache as dc
import atexit
import json
import tempfile
import pdfplumber
import pandas as pd

# All LlamaIndex tools needed...LLM, memory, roles, etc
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.llms import ChatMessage, MessageRole
from utils import process_user_query, get_colored_text
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec
from llama_index.core.indices import VectorStoreIndex
from llama_index.core import Document
from llama_index.embeddings.google import GeminiEmbedding
import torch

# Autogen agents
from autogen import ConversableAgent
from autogen.coding import LocalCommandLineCodeExecutor

# Summarizer of history
from transformers import pipeline

## Constants/Configurations
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
google_api_key = "AIzaSyD939q3PbECaSJO1IAzRbmpqlREgJteLKg"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

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
if 'sec_embedder' not in st.session_state:
    st.session_state.sec_embedder = GeminiEmbedding(model_name='models/embedding-001')
if 'sec_store' not in st.session_state:
    # st.session_state.sec_store = VectorStoreIndex(embed_model=st.session_state.sec_embedder).from_documents(st.session_state.uploaded_docs)
    st.session_state.sec_store = VectorStoreIndex.from_documents(documents=st.session_state.uploaded_docs, **{'embed_model': st.session_state.sec_embedder})

# Tavily search tool setup
tavily_api_key = "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"
if not os.environ.get('TAVILY_API_KEY'):
    os.environ['TAVILY_API_KEY'] = tavily_api_key
search_tool = TavilyToolSpec(api_key=tavily_api_key)
search_tool = search_tool.to_tool_list()

# Initializing a simple agent with a search tool
agent = ReActAgent.from_tools(tools=search_tool, llm=gemini_model)

agent_system_prompt = "Respond concisely and accurately, using the conversation provided and the context specified in the query. The user may reference documents they provided, which will be given to you as context."
executor = LocalCommandLineCodeExecutor(work_dir=temp_dir.name)
auto_agent = ConversableAgent(name="code_executor", human_input_mode="NEVER", system_message=agent_system_prompt,
                                llm_config={"config_list": [{"model": "gemini-1.5-flash", "temperature": 0.5, "api_key": os.environ.get("GOOGLE_API_KEY"), "api_type": "google"}]})

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

# Document uploader for including document text in the chat prompt
with st.sidebar:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload a .txt, .json, .csv or .pdf document", type=["txt", "json", "pdf", "csv"])
    attached_text = ""

    if uploaded_file:
        file_size = uploaded_file.size

        if uploaded_file.type == "application/pdf":
            if uploaded_file.name in document_cache:
                attached_text = document_cache[uploaded_file.name]
                st.success(f"Loaded cached document: {uploaded_file.name}")
            else:
                attached_pdf_text = ""
                with pdfplumber.open(uploaded_file) as pdf:
                    for page in pdf.pages:
                        non_table_text = extract_non_table_text(page)
                        attached_pdf_text += non_table_text + "\n"
                        # Modify to handle multiple tables effectively
                        tables = page.extract_tables()
                        if tables:
                            attached_pdf_text += "\n--- Table Data ---\n"
                            for table in tables:
                                for row in table:
                                    attached_pdf_text += " | ".join(cell if cell else "" for cell in row) + "\n"
                            attached_pdf_text += "--- End of Table ---\n"     
                attached_text = attached_pdf_text
                st.success("PDF parsed!")
            
        elif uploaded_file.type == "text/plain":
            attached_text = uploaded_file.read().decode("utf-8")
            st.success("Text document loaded!")
        elif uploaded_file.type == "application/json":
            attached_text = json.dumps(json.load(uploaded_file), indent=2)
            st.success("JSON document loaded!")
        elif uploaded_file.type == "text/csv":
            # Size threshold in bytes (e.g., 5 MB)
            
            if file_size <= SMALL_FILE_SIZE_THRESHOLD:
                # Directly parse smaller CSV files
                df = pd.read_csv(uploaded_file)
                attached_text = df.to_string(index=False)  # Convert DataFrame to string
                st.success("Small CSV loaded and parsed!")
            else:
                # Process larger CSV files with derived insights
                df = pd.read_csv(uploaded_file)
                
                # Generate summary of the data
                summary_info = f"""
                File Size: {file_size / (1024 * 1024):.2f} MB
                Number of Rows: {df.shape[0]}
                Number of Columns: {df.shape[1]}
                Columns: {', '.join(df.columns)}
                """
                attached_text = summary_info
                
                st.success("Large CSV loaded! Showing summary:")
                st.text(summary_info)
                
                st.write("Preview of Data:")
                st.dataframe(df.head(10))  # Show top 10 rows

                st.session_state.df = df

        # Add the parsed text as a Document for secondary vector store
        doc = Document(text=attached_text, metadata={"filename": uploaded_file.name})
        st.session_state.uploaded_docs.append(doc)

        # Update secondary vector store with new document
        st.session_state.uploaded_docs.append(doc)
        st.session_state.sec_store = VectorStoreIndex.from_documents(st.session_state.uploaded_docs, **{'embed_model': st.session_state.sec_embedder})
        st.success(f"Document '{uploaded_file.name}' added to the secondary vector store!")

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

        full_prompt = f"{user_input}\n\nAttached Document: {attached_text}" if attached_text else user_input
        with st.spinner("Analyzing user query..."):
            # contextualized_prompt = contextualize_prompt(user_input)
            document_txt = attached_text if attached_text else ""
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
                
                if attached_text:
                    # Modify the last message in the session state to include document context
                    last_message = st.session_state.chat_messages[-1]
                    last_message.content = f"{last_message.content}\n\nAttached Document Context:\n{attached_text}"

                formatted_messages = [
                    {"role": "user" if msg.role == MessageRole.USER else "assistant", "content": msg.content}
                    for msg in st.session_state.chat_messages
                ]
                auto_reply = auto_agent.generate_reply(messages=formatted_messages)

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
                print(combined_context_text)
                # Prepare the input for the ConversableAgent
                formatted_messages = [
                    {"role": "user" if msg.role == MessageRole.USER else "assistant", "content": msg.content}
                    for msg in st.session_state.chat_messages[:-1]
                ]

                # Include context and the reformulated query
                formatted_messages.append({"role": "user", "content": f"{contextualized_prompt}\nContext:\n{context_text}"})

                # Generate a response using the ConversableAgent
                response = auto_agent.generate_reply(messages=formatted_messages)
                assistant_response = response['content']

                # Display the response
                st.markdown(assistant_response)
                st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response))
                st.session_state.chat_messages_display.append(ChatMessage(role=MessageRole.ASSISTANT, content=assistant_response))

            else:
                print("Invalid response type detected:", query_type)
                st.error("Invalid response type detected. Please try again.")

    print(st.session_state.chat_messages)
    st.rerun()