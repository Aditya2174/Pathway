# Necessary imports
import streamlit as st
import os
from llama_index.llms.gemini import Gemini
from llama_index.retrievers.pathway import PathwayRetriever
from llama_index.core.prompts import ChatPromptTemplate
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.core.agent import ReActAgent
from llama_index.tools.tavily_research import TavilyToolSpec
import pdfplumber
import json

# Initialize a Gemini-1.5-Flash model with LlamaIndex
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8755

retriever = PathwayRetriever(host=PATHWAY_HOST, port=PATHWAY_PORT)

# Initialize memory if not already in session state
if 'memory' not in st.session_state:
    st.session_state.memory = ChatMemoryBuffer(token_limit=10000000)

# Tavily search tool setup
tavily_api_key = "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"
if not os.environ.get('TAVILY_API_KEY'):
    os.environ['TAVILY_API_KEY'] = tavily_api_key
search_tool = TavilyToolSpec(api_key=tavily_api_key)
search_tool = search_tool.to_tool_list()

reformat_content="""You are an intelligent AI assistant. You will be provided with a user query and the chat history between the user and the chatbot. You will have to identify the type of the query, and give your output according to the following rules:
- If the context of query is similar to the chat history, reformulate the query based on the chat history. Ensure that the reformulated query is as detailed and contextually rich as possible. Respond with query type as 'reformulation' and output as the reformulated query.
- If the query is general and unrelated to the chat history and doesn't require any particular information to be answered, then respond with query type as 'general' and output as a polite response to the query.
- If the query is completely unrelated to the chat history and require additional information to be answered, respond with query type as 'unrelated' and output the gramatically corrected query.

Examples are provided below.

Example 1:
Chat History:
user: Who discovered the laws of motion?
ai: Isaac Newton
User Query: Tell me more about him
Query Type: reformulation
Output: Tell me more about Isaac Newton who discovered the laws of motion.

Example 2:
Chat History:

User Query: How are you?
Query Type: general
Output: I am doing well, thank you for asking. How can I help you today?

Example 3:
Chat History:
user: Who discovered the laws of motion?
ai: Isaac Newton
User Query: Waht is the capital of France?
Query Type: unrelated
Output: What is the capital of France?

"""

# Initializing a simple agent with a search tool
agent = ReActAgent.from_tools(tools=search_tool, llm=gemini_model)

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Answer the question with whatever information is available, and if more is needed, use the tool for a web search.\
                     Use the context provided if any is available. "
                "If there is not enough information still, ask the user for clarification."
            ),
        ),
    ]

# Function to reformat user's prompt with history as context
def contextualize_prompt(user_prompt):
    context_template = [ChatMessage(role=MessageRole.SYSTEM, content=reformat_content)]
    context_template.extend(st.session_state.chat_messages[1:])
    context_template.append(ChatMessage(role=MessageRole.USER, content=user_prompt))

    reformat_prompt = ChatPromptTemplate.from_messages(context_template).format_messages()
    response = gemini_model.chat(messages=reformat_prompt).message.content

    return response

# Identifying whether retrieval is needed
def determine_response_type(user_prompt):
    analysis_template = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Analyze if the prompt can be answered using chat history alone or if it requires document retrieval or a web search."
                "If chat history is sufficient, respond with 'direct answer'."
                "If additional documents or web searches are needed, respond with 'needs more context'."
            ),
        ),
        ChatMessage(role=MessageRole.USER, content=user_prompt),
    ]

    analysis_template = ChatPromptTemplate.from_messages(analysis_template).format_messages()
    analysis_template.extend(st.session_state.chat_messages[1:])
    response = gemini_model.chat(analysis_template).message.content.lower()

    if 'direct answer' in response:
        return "direct_answer"
    else:
        return "needs_more_context"

def extract_non_table_text(page):
    non_table_text = ""
    # Extract tables to get their bounding boxes
    tables = page.extract_tables()
    table_bboxes = [table.bbox for table in page.find_tables()]

    # Extract text boxes, filtering out any within table bounding boxes
    for word in page.extract_words():
        word_bbox = (word['x0'], word['top'], word['x1'], word['bottom'])
        # Check if the word bounding box overlaps with any table bounding box
        if not any((word_bbox[0] >= bbox[0] and word_bbox[1] >= bbox[1] and
                    word_bbox[2] <= bbox[2] and word_bbox[3] <= bbox[3]) for bbox in table_bboxes):
            non_table_text += word['text'] + " "

    return non_table_text.strip()

with st.sidebar:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload a .txt, .json, or .pdf document", type=["txt", "json", "pdf"])
    attached_text = ""

    if uploaded_file:
        if uploaded_file.type == "application/pdf":
            attached_pdf_text = ""
            with pdfplumber.open(uploaded_file) as pdf:
                for page in pdf.pages:
                    # Extract non-table text using bounding box filtering
                    non_table_text = extract_non_table_text(page)
                    attached_pdf_text += non_table_text + "\n"

                    # Optionally, add structured tables to attached_pdf_text
                    tables = page.extract_tables()
                    if tables:
                        attached_pdf_text += "\n--- Table Data ---\n"
                        for table in tables:
                            for row in table:
                                attached_pdf_text += " | ".join(cell if cell else "" for cell in row) + "\n"
                        attached_pdf_text += "--- End of Table ---\n"

            attached_text = attached_pdf_text
            st.success("PDF parsed with non-table text extracted!")

        elif uploaded_file.type == "text/plain":
            attached_text = uploaded_file.read().decode("utf-8")
            st.success("Text document loaded for context!")

        elif uploaded_file.type == "application/json":
            attached_text = json.dumps(json.load(uploaded_file), indent=2)
            st.success("JSON document loaded for context!")

# Display the chat history
for chat in st.session_state.chat_messages[1:]:
    role = "assistant" if chat.role == MessageRole.ASSISTANT else "user"
    st.chat_message(role).write(chat.content)

# Chat input for user to enter prompt
if user_input := st.chat_input("Enter your chat prompt:"):
    # Add the user's message to chat history immediately
    print(user_input)
    st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))
    st.chat_message("user").write(user_input)

    # Include document text in the prompt if checked
    full_prompt = f"Prompt: {user_input}\nAttached context retrieved from user's documents: {attached_text}"
    # Process user prompt
    contextualized_prompt = contextualize_prompt(full_prompt)
    print("contextualized prompt: "  + contextualized_prompt)
    response_type = determine_response_type(contextualized_prompt)
    
    # Retrieve answer based on response type
    if response_type == "direct_answer":
        print("Direct answer")
        
        final_prompt = f"Prompt: {contextualized_prompt}"
        # Add the user's message to chat history
        # st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=final_prompt))
        input_prompt = st.session_state.chat_messages[:-1]
        input_prompt.append(ChatMessage(role=MessageRole.USER, content=final_prompt))
        # Get the model's response
        response = gemini_model.chat(ChatPromptTemplate.from_messages(input_prompt)).format_messages().message.content
        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))
        st.chat_message("assistant").write(response)
    else:
        print("Needs more context")
        context = retriever.retrieve(contextualized_prompt)
        context_text = "\n".join([doc.text for doc in context])
        final_prompt = f"Prompt: {contextualized_prompt}\nContext: {context_text}"
        
        # Add the user's message to chat history
        # st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=final_prompt))
        input_prompt = st.session_state.chat_messages[:-1]
        input_prompt.append(ChatMessage(role=MessageRole.USER, content=final_prompt))
        # Get the agent's response
        response = agent.chat(message=final_prompt, chat_history=input_prompt).response

        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))
        st.chat_message("assistant").write(response)
