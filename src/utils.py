from llama_index.core import PromptTemplate
from datetime import datetime
from llama_index.core.llms import ChatMessage, MessageRole
from prompts import reformulation_type_prompt

def process_user_query(llm, chat_history, user_query, document):
    current_date = datetime.now().strftime("%Y-%m-%d")

    chat_history_li = []
    for chat in chat_history:
        if chat.role == MessageRole.USER:
            chat_history_li.append(f"user: {chat.content}")
        else:
            chat_history_li.append(f"ai: {chat.content}")

    chat_history_str = '\n'.join(chat_history_li)

    prompt = PromptTemplate((reformulation_type_prompt))
    prompt = reformulation_type_prompt.format(current_date=current_date, chat_history=chat_history_str, user_query=user_query, document=document)
    resp = llm.complete(prompt).text

    resp_li = resp.split('\n')
    resp_li = [x for x in resp_li if x != '']
    query_type = resp_li[0].split('Query Type:')[-1].strip()
    output = resp_li[1].split('Output:')[-1].strip()
    return query_type, output

def get_colored_text(text, color = 'green'):
    txt = f"""Reformulated to: <span style="color: {color}; font-size: 14px;">{text}</span>"""
    return txt