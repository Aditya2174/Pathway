import os
import pandas as pd
import time
from langchain_community.vectorstores.pathway import PathwayVectorClient
from langchain_core.documents import Document
from langchain.prompts.chat import ChatPromptTemplate
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent

# Set up Google Gemini model with LangChain
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"  # Remember to replace with your API key
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key
model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8755
vector_client = PathwayVectorClient(host=PATHWAY_HOST, port=PATHWAY_PORT)

# Initialize memory
memory = ConversationBufferMemory(memory_key="chat_history", input_key="user_prompt", return_messages=True)

# Creating a search tool and configuring API settings
tavily_api_key = "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"
if not os.environ.get('TAVILY_API_KEY'):
    os.environ['TAVILY_API_KEY'] = tavily_api_key

search = TavilySearchResults(max_results=5, include_answer=True)
tools = [search]

# Creating a simple agent to handle tool calling
agent = create_react_agent(model=model, tools=tools)

def can_answer(user_prompt, attached_doc_text):
    template = [
        SystemMessage(content="Analyze if the prompt is a follow-up question or a standalone question. "
                              "If it's a follow-up that needs prior context, respond with 'needs more context'. "
                              "If the prompt can be answered directly, respond with 'direct answer'. "
                              "If it's an independent question, respond with 'standalone question'."),
        MessagesPlaceholder("chat_history"),
        HumanMessage(content=user_prompt)
    ]
    
    final_prompt = ChatPromptTemplate.from_messages(template).format_prompt(**{"chat_history": memory.chat_memory.messages})
    response = model.invoke(final_prompt).content.lower()

    return "needs_more_context"
    if 'direct answer' in response:
        return "direct_answer"
    elif 'needs more context' in response:
        return "needs_more_context"
    else:
        return "standalone_question"

def evaluate_pipeline(data_path, checkpoint_path='checkpoint.txt', results_path='evaluation_results.csv'):
    df = pd.read_csv(data_path)
    results = []

    # Load checkpoint
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path, 'r') as f:
            start_index = int(f.read().strip())
    else:
        start_index = 0

    # Load previous results if they exist
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        results = results_df.to_dict('records')

    counter = 0

    for index, row in df.iterrows():
        if index < start_index:
            continue

        user_prompt = row['question']
        attached_doc_text = row['context']
        print(f"Evaluating row {index}...")
        print(f"User Prompt: {user_prompt}")
        response_type = can_answer(user_prompt, attached_doc_text or "")

        print(f"Response Type: {response_type}")
        if "direct answer" in response_type.lower():
            full_prompt = user_prompt
            if attached_doc_text:
                full_prompt += f"\n\nAttached Document: {attached_doc_text}"

            template = [
                SystemMessage(content="Answer the user's prompt based on the chat history and the attached document (if any)."),
                MessagesPlaceholder("chat_history"),
                HumanMessage(content=full_prompt)
            ]
            chat_prompt_template = ChatPromptTemplate.from_messages(template)

            formatted_prompt = chat_prompt_template.format_prompt(**{"chat_history": memory.chat_memory.messages})
            response = model.invoke(formatted_prompt).content
            final_response = agent.stream(formatted_prompt)
            stream_responses = [message for message in final_response]

            response_content = stream_responses[-1]['agent']['messages'][0].content

            # Update memory with the new user message and context
            memory.chat_memory.add_message(HumanMessage(content=user_prompt))
            memory.chat_memory.add_message(AIMessage(content=response_content))

        elif "needs_more_context" in response_type.lower():
            retriever = vector_client.as_retriever()
            context = retriever.invoke(user_prompt)
            
            content_list = [doc.page_content for doc in context]
            retrieved = content_list
            full_prompt = f"{user_prompt}\n\nContext: {content_list}"
            if attached_doc_text:
                full_prompt += f"\n\nAttached Document: {attached_doc_text}"

            template = [
                SystemMessage(content="Answer using the context provided."),
                MessagesPlaceholder("chat_history"),
                HumanMessage(content=full_prompt)
            ]
            chat_prompt_template = ChatPromptTemplate.from_messages(template)

            formatted_prompt = chat_prompt_template.format_prompt(**{"chat_history": memory.chat_memory.messages})
            final_response = agent.stream(formatted_prompt)
            stream_responses = [message for message in final_response]

            response_content = stream_responses[-1]['agent']['messages'][0].content

            memory.chat_memory.add_message(HumanMessage(content=user_prompt))
            memory.chat_memory.add_message(AIMessage(content=response_content))

        else:  # standalone_question
            full_prompt = user_prompt
            if attached_doc_text:
                full_prompt += f"\n\nAttached Document: {attached_doc_text}"

            template = [
                SystemMessage(content="Answer the user's prompt as a standalone question."),
                HumanMessage(content=full_prompt)
            ]
            chat_prompt_template = ChatPromptTemplate.from_messages(template)

            formatted_prompt = chat_prompt_template.format_prompt(**{"chat_history": memory.chat_memory.messages})
            final_response = agent.stream(formatted_prompt)
            stream_responses = [message for message in final_response]

            response_content = stream_responses[-1]['agent']['messages'][0].content

            memory.chat_memory.add_message(HumanMessage(content=user_prompt))
            memory.chat_memory.add_message(AIMessage(content=response_content))

        results.append({
            'question': user_prompt,
            'context': retrieved,
            'response': response_content
        })
        print(f"Response: {response_content}")
        print(f"Row {index} evaluated successfully.")
        
        counter += 1
        if counter % 2 == 0:
            result_df = pd.DataFrame(results)
            result_df.to_csv(results_path, index=False, mode='a', header=not os.path.exists(results_path))
            print(f"Intermediate results saved to '{results_path}'.")

            # Save checkpoint
            with open(checkpoint_path, 'w') as f:
                f.write(str(index + 1))

            print(f"Checkpoint saved at row {index + 1}.")
            time.sleep(1)  # Sleep for 1 second after every 2 questions

    result_df = pd.DataFrame(results)
    result_df.to_csv(results_path, index=False, mode='a', header=not os.path.exists(results_path))
    print(f"Evaluation completed. Results saved to '{results_path}'.")

    # Remove checkpoint file after completion
    if os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)

# Example usage
data_path = 'data.csv'
evaluate_pipeline(data_path)
