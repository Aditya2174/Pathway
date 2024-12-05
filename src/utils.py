from llama_index.core import PromptTemplate
from datetime import datetime
from llama_index.core.llms import ChatMessage, MessageRole
from prompts import (
    reformulation_type_prompt,
    reformulation_type_prompt_no_doc,
    query_classification_prompt,
    query_classification_prompt_no_doc
)
from llama_index.llms.gemini import Gemini
import tiktoken
import pdfplumber
import json
import os
import pandas as pd
import streamlit as st
import google.generativeai as genai
from typing import Tuple
import time
if 'tiktoken_tokenizer' not in st.session_state:
    st.session_state.tiktoken_tokenizer = tiktoken.encoding_for_model('gpt-4')
from prompts import reformulation_type_prompt

def get_history_str(chat_history):
    chat_history_li = []
    for chat in chat_history:
        if chat.role == MessageRole.USER:
            chat_history_li.append(f"user: {chat.content}")
        else:
            chat_history_li.append(f"ai: {chat.content}")

    chat_history_str = '\n'.join(chat_history_li)
    return chat_history_str

def build_prompt(user_query, chat_history = [], retrieved_context = [], doc_context = [], search_results = []):
    final_prompt = f"# Query: {user_query}\n"
    if chat_history != []:
        chat_history_str = get_history_str(chat_history)
        final_prompt += f"# Chat History:\n{chat_history_str}\n"
    if doc_context != []:
        final_prompt += "# User Document Context:\n"
        for i, context in enumerate(doc_context):
            final_prompt += f"## User Document {i+1}:\n{context}\n"
    if retrieved_context != []:
        final_prompt += "# Retrieved Document Context:\n"
        for i, context in enumerate(retrieved_context):
            final_prompt += f"## Retrieved Document {i+1}:\n{context}\n"
    if search_results != []:
        final_prompt += "# Web Search Results:\n"
        for i, result in enumerate(search_results):
            final_prompt += f"## Web Search Result {i+1}:\n{result}\n"
    
    return final_prompt

def process_user_query(llm : Gemini , chat_history, user_query, document) -> Tuple[str, str, int]:
    current_date = datetime.now().strftime("%Y-%m-%d")
    chat_history_li = []
    for chat in chat_history:
        if chat.role == MessageRole.USER:
            chat_history_li.append(f"user: {chat.content}")
        else:
            chat_history_li.append(f"ai: {chat.content}")

    chat_history_str = '\n'.join(chat_history_li)

    if document is not None:
        prompt = PromptTemplate((reformulation_type_prompt))
        prompt = prompt.format(current_date=current_date, chat_history=chat_history_str, user_query=user_query, document=document)
    else:
        prompt = PromptTemplate((reformulation_type_prompt_no_doc))
        prompt = prompt.format(current_date=current_date, chat_history=chat_history_str, user_query=user_query)

    result = llm.complete(prompt)
    cost = result.raw['usage_metadata']['total_token_count']
    resp = result.text

    resp_li = resp.split('\n')
    resp_li = [x for x in resp_li if x != '']
    query_type = resp_li[0].split('Query Type:')[-1].strip()
    output = resp_li[1].split('Output:')[-1].strip()
    return query_type, output,cost

def get_colored_text(text, color = 'green'):
    txt = f"""Reformulated to: <span style="color: {color}; font-size: 14px;">{text}</span>"""
    return txt

def get_num_tokens(txt):
    num_tokens = len(st.session_state.tiktoken_tokenizer.encode(txt))
    return num_tokens

def classify_query(llm, user_query, query_type):
    if 'direct' in query_type:
        # Document is provided
        prompt = PromptTemplate((query_classification_prompt))
        prompt = prompt.format(user_query=user_query)
    else:
        # Document is not provided
        prompt = PromptTemplate((query_classification_prompt_no_doc))
        prompt = prompt.format(user_query=user_query)

    result = llm.complete(prompt)
    cost = result.raw['usage_metadata']['total_token_count']
    resp = result.text
    resp_li = resp.split('\n')
    resp_li = [x for x in resp_li if x != '']
    query_type = resp_li[0].strip()
    return query_type

def hyde(input_query, model) -> Tuple[str, int]:
    query = "Expand the given query into a document for retrieval. Do not add any numerical information or ask for clarification or make unnecessary changes.\n"+input_query
    result = model.complete(query, generation_config={'max_output_tokens':200,"temperature":0.2})
    query_doc = result.text
    cost = result.raw['usage_metadata']['total_token_count']
    
    return f"{input_query}\n {query_doc}", cost
    
def is_plot_in_response(response):
    if ('plots/image_' in response) and ('.png' in response):
        return True
    return False

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

def read_file_from_cache_or_parse(uploaded_file, cache, file_size_threshold, large_file_directory):
    """Reads a file from cache if available, otherwise parses it."""
    file_name = uploaded_file.name
    file_type = uploaded_file.type

    # Check if the file is large and needs to be stored in the large file directory
    if uploaded_file.size > file_size_threshold:
        large_file_path = os.path.join(large_file_directory, file_name)
        with open(large_file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.warning(f"Large file '{file_name}' saved to directory. Only partial content is being processed.")

        # Process limited content based on file type
        if file_type == "application/pdf":
            attached_text = ""
            with pdfplumber.open(large_file_path) as pdf:
                for page in pdf.pages[:2]:  # Limit to the first 2 pages
                    non_table_text = extract_non_table_text(page)
                    attached_text += non_table_text + "\n"

        elif file_type == "text/plain":
            with open(large_file_path, "r", encoding="utf-8") as f:
                attached_text = f.read(1000)  # Limit to the first 1000 characters

        elif file_type == "application/json":
            with open(large_file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            attached_text = json.dumps(data, indent=2)[:1000]  # Limit JSON preview to 1000 characters

        elif file_type == "text/csv":
            df = pd.read_csv(large_file_path)
            preview = df.head()
            attached_text = f"""
            File Size: {uploaded_file.size / (1024 * 1024):.2f} MB
            Number of Rows: {df.shape[0]}
            Number of Columns: {df.shape[1]}
            Columns: {', '.join(df.columns)}
            Preview: {preview.to_string()}
            """
            st.write("Preview of Data:")
            st.dataframe(df.head(10))
        else:
            attached_text = f"File type '{file_type}' not supported for partial processing."

        # Add the limited content to the cache
        cache[file_name] = attached_text
        return attached_text

    # If the file is not large, process normally
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

def evaluate_sufficiency(gemini_model, context_text: str, query: str, total_cost :int) -> Tuple[bool, int]:
        """Use Gemini model to evaluate if the retrieved context suffices to answer the query."""
        evaluation_prompt = (
            f"Question: {query}\n\n"
            f"Context:\n{context_text}\n\n"
            "Does the context provide sufficient information to fully answer the question?"
            "Respond with 'Yes' or 'No' only."
        )
        time.sleep(1) # To avoid rate limiting
        # result = gemini_model.generate_response(evaluation_prompt)
        result = gemini_model.chat([ChatMessage(content=evaluation_prompt, role=MessageRole.USER)])
        cost = result.raw['usage_metadata']['total_token_count']
        if 'no' in result.message.content.lower():
            return False, cost
        return True, cost