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
<<<<<<< Updated upstream
=======
import pypdf
>>>>>>> Stashed changes
import pdfplumber
import json

# Initialize a Gemini-1.5-Flash model with LlamaIndex
google_api_key = "AIzaSyAw786vp_FhAWxi9vce2IoHon53sGxeCdk"
if not os.environ.get('GOOGLE_API_KEY'):
    os.environ['GOOGLE_API_KEY'] = google_api_key

gemini_model = Gemini(model="models/gemini-1.5-flash", api_key=google_api_key)
gemini_pro_model = Gemini(model='models/gemini-1.5-pro', api_key=google_api_key)

# Pathway server configuration
PATHWAY_HOST = "127.0.0.1"
PATHWAY_PORT = 8756

retriever = PathwayRetriever(host=PATHWAY_HOST, port=PATHWAY_PORT)

# Initialize memory if not already in session state
if 'context' not in st.session_state:
    st.session_state.context = ChatMemoryBuffer(token_limit=500000)

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
User Query: What is the capital of France?
Query Type: unrelated
Output: What is the capital of France?

Only return the Query Type and Output. Do not answer the User Query. Only refactor the query to better represent the chat history and context.

"""

response_tpye_content = """ 
You are an intelligent AI assistant. You will be provided with a user query and the chat history between the user and the chatbot. You will have to identify how to answer the query, give your output according to following rules:
- If the history has enough data for answering the query and you can answer it confidently wihtout relying on external context. Respond with "direct"
-Else if you are doubtful about your ability to answer the question based on only the provided context, respond with "context"

Follow the given examples for reference

Example 1:
Chat History:
The water molecule is made up of 2 atoms of hydrogen bonded with 1 oxygen atom.
User Query: Tell me about the structure of the water molecule.
Output: direct

Example 2:
Chat History:
The Super Bowl is the annual league championship game of the National Football League (NFL) of the United States.
User Query: When is the next super bowl happening.
Output: context

Only return the Output. Do not answer the User Query. Only answer from context or direct, do not return empty or none or anything else.

"""

# Initializing a simple agent with a search tool
agent = ReActAgent.from_tools(tools=search_tool, llm=gemini_pro_model)

if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Answer the question with whatever information is available, and if more is needed, use the tool for a web search.\
                     Use the context provided if any is available. "
                "If there is not enough information still, ask the user for clarification. Make sure to give clear and concise answers wihout any ambiguity."
            ),
        ),
    ]

if 'reformulated_messages' not in st.session_state:
    st.session_state.reformulated_messages = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=(
                "Answer the question with whatever information is available, and if more is needed, use the tool for a web search.\
                     Use the context provided if any is available. "
                "If there is not enough information still, ask the user for clarification. Make sure to give clear and concise answers wihout any ambiguity."
            ),
        ),
    ]


# Function to reformat user's prompt with history as context
def contextualize_prompt(user_prompt):
    context_template = [ChatMessage(role=MessageRole.SYSTEM, content=reformat_content)]
    context_template.extend(st.session_state.reformulated_messages[1:])
    context_template.append(ChatMessage(role=MessageRole.USER, content=user_prompt))
    context_template.append(ChatMessage(role=MessageRole.USER, content=f"Use the following context to answer the above chat:\n{st.session_state.context}"))

    reformat_prompt = ChatPromptTemplate.from_messages(context_template).format_messages()
    response = gemini_model.chat(messages=reformat_prompt).message.content

    return response

# Identifying whether retrieval is needed
def determine_response_type(user_prompt):
    analysis_template = [
        ChatMessage(
            role=MessageRole.SYSTEM,
            content=response_tpye_content,
        )]
    analysis_template.extend(st.session_state.reformulated_messages[1:])
    # analysis_template.append(ChatMessage(role=MessageRole.USER, content=user_prompt))
    analysis_template.append(ChatMessage(role=MessageRole.USER, content=f"Use the following context to answer the above chat:\n{st.session_state.context}"))

    analysis_template = ChatPromptTemplate.from_messages(analysis_template).format_messages()
    response = gemini_model.chat(analysis_template).message.content.lower()

    if 'direct' in response:
        return "direct_answer"
    elif 'context' in response:
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

<<<<<<< Updated upstream
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

=======
>>>>>>> Stashed changes
# Document uploader for including document text in the chat prompt
with st.sidebar:
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload a .txt, .json, or .pdf document", type=["txt", "json", "pdf"])
    attached_text = ""

<<<<<<< Updated upstream
    st.subheader("Upload Document")
    uploaded_file = st.file_uploader("Upload a .txt, .json, or .pdf document", type=["txt", "json", "pdf"])
    attached_text = ""

=======
>>>>>>> Stashed changes
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
<<<<<<< Updated upstream
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
=======
>>>>>>> Stashed changes

# Display the chat history
for chat in st.session_state.chat_messages:
    if chat.role == 'system':
        continue
    role = "assistant" if chat.role == MessageRole.ASSISTANT else "user"
    st.chat_message(role).write(chat.content)

# Chat input for user to enter prompt
if user_input := st.chat_input("Enter your chat prompt:"):
    # Add the user's message to chat history immediately
    print("User Input: " + user_input)
    st.chat_message("user").write(user_input)
    st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=user_input))

    # Include document text in the prompt if checked
    
    # full_prompt = f"{user_input}\n\nAttached Document: {attached_doc_text}" if attached_doc_text else user_input
    # full_prompt = f"{full_prompt}\nAttached PDF: {attached_pdf_text}" if attached_pdf_text else full_prompt
    # print(attached_pdf_text[:50])
    # Process user prompt
    
    contextualized_prompt_response = contextualize_prompt(user_input)
    contextualized_prompt = contextualized_prompt_response.split('\n')[1].split('Output: ')[1]
    print("contextualized prompt response:\n"  + contextualized_prompt_response)
    query_type = contextualized_prompt_response.split('\n')[0].split('Query Type: ')[1].lower()
    st.session_state.reformulated_messages.append(ChatMessage(role=MessageRole.USER, content=contextualized_prompt))
    if(query_type=='general'):
        st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=contextualized_prompt))
        st.session_state.reformulated_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=contextualized_prompt))
        st.chat_message("assistant").write(contextualized_prompt)
        # break
    else:
        response_type = determine_response_type(contextualized_prompt)
        print("Response type: " + response_type)
        
        response_type = 'needs_more_context'
<<<<<<< Updated upstream
        response_type = 'needs_more_context'
=======
>>>>>>> Stashed changes
        # Retrieve answer based on response type
        if response_type == "direct_answer":
            # context = retriever.retrieve(contextualized_prompt)
            # context_text = "\n".join([doc.text for doc in context])
            # final_prompt = f"Prompt: {contextualized_prompt}\nContext: {context_text}" 
<<<<<<< Updated upstream
            # final_prompt = f"Prompt: {contextualized_prompt}\nContext: {context_text}" 
=======
>>>>>>> Stashed changes
            
            # final_prompt = f"Prompt: {contextualized_prompt}"
            # Add the user's message to chat history
            # st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=final_prompt))
            final_prompt = st.session_state.reformulated_messages+ [ChatMessage(role=MessageRole.USER, content=f"\n Use the following context to answer the above chat:\n {st.session_state.context}")]
            
            # Get the model's response
            final_prompt = ChatPromptTemplate.from_messages(final_prompt).format_messages()
            response = gemini_pro_model.chat(final_prompt).message.content
<<<<<<< Updated upstream
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))
            st.session_state.reformulated_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))
            st.chat_message("assistant").write(response)

        elif response_type == "needs_more_context":
            # print("Needs more context")
            context = retriever.retrieve(contextualized_prompt)
            print(context)
            print(context[0])
            context_text = "\n".join([doc.text for doc in context])
            print("Retrieved context:\n"+context_text)
            # final_prompt = f"Prompt: {contextualized_prompt}\nContext: {context_text}"
            final_prompt = f"User Query: {contextualized_prompt}\nUse the following context to answer the above chat:\n{context_text}"
            # ChatMessage(role=MessageRole.USER, content=f"Use the following context to answer the above chat:\n {st.session_state.context}\n{context_text}")
            
            # Add the user's message to chat history
            # st.session_state.chat_messages.append(ChatMessage(role=MessageRole.USER, content=final_prompt))
            
            # Get the agent's response
            response = agent.chat(message=final_prompt, chat_history=st.session_state.reformulated_messages[:-1]).response
            st.session_state.chat_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))
            st.session_state.reformulated_messages.append(ChatMessage(role=MessageRole.ASSISTANT, content=response))
            st.chat_message("assistant").write(response)
=======
            st.session_state.chat_messag
>>>>>>> Stashed changes
