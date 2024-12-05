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
import streamlit as st
import google.generativeai as genai
from typing import Tuple
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

def hyde(query, model) -> Tuple[str, int]:
    query = "Expand the given query into a document for retrieval. Do not add any numerical information or ask for clarification or make unnecessary changes.\n"+query
    print("start:"+query)
        # print("start:"+query)
    result = model.complete(query, generation_config={'max_output_tokens':200,"temperature":0.2})
    query_doc = result.text
    cost = result.raw['usage_metadata']['total_token_count']
    
    print("end:"+query_doc)
    return query_doc+query, cost
    
