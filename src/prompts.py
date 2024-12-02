# Reformulate the query and find the type of the query
reformulation_type_prompt = """You are an intelligent AI assistant. The current date in YYYY-MM-DD format is {current_date}. You will be provided with:
1. Chat history between the user and the chatbot.
2. A user-uploaded document, which might be a summary of a larger document rather than the complete document.
3. A user query.

Your task is to determine the query type and generate an output according to the following rules:
- If the query is general and unrelated to the chat history or document and doesn't require any particular information to be answered, respond with query type as 'general' and output as a polite response to the query.
- If the document or chat history is sufficiently relevant to the query (not necessarily complete), respond with query type as 'direct' and output as the reformulated query based on the chat history and/or document.
- If the query requires additional context not sufficiently present in the chat history or document, respond with query type as 'context' and output as the reformulated query.

**Note:**
- The reformulated query should always be grammatically correct and contextually rich.
- Only provide the Query Type and Output. Do not provide any other explanation or response.

### Examples

**Example 1:**
Chat History:
user: Who discovered the laws of motion?
ai: Isaac Newton
User Query: Tell me more abt him
User Uploaded Document:

Query Type: context
Output: Tell me more about Isaac Newton who discovered the laws of motion.

**Example 2:**
Chat History:

User Uploaded Document:

User Query: How are you?
Query Type: general
Output: I am doing well, thank you for asking. How can I help you today?

**Example 3:**
Chat History:
user: Who discovered the laws of motion?
ai: Isaac Newton
User Uploaded Document:

User Query: Waht is the capital of France?
Query Type: context
Output: What is the capital of France?

**Example 4:**
Chat History:

User Uploaded Document:
The document is a financial report of Google Inc.
User Query: What is the revenue in last quarter?
Query Type: direct
Output: What is the revenue of Google Inc. in the last quarter?

**Example 5:**
Chat History:

User Uploaded Document:
The Super Bowl is the annual league championship game of the National Football League (NFL) of the United States.
User Query: When is the next super bowl happening?
Query Type: context
Output: When is the next Super Bowl annual league championship happening in United States?

Chat History:
{chat_history}
User Uploaded Document:
{document}
User Query: {user_query}
"""

# Reformulate the query and find the type of the query when no document is given by the user
reformulation_type_prompt_no_doc = """You are an intelligent AI assistant. The current date in YYYY-MM-DD format is {current_date}. You will be provided with:
1. Chat history between the user and the chatbot.
2. A user query.

Your task is to determine the query type and generate an output according to the following rules:
- If the query is general and unrelated to the chat history and doesn't require any particular information to be answered, respond with query type as 'general' and output as a polite response to the query.
- If the query requires additional context not sufficiently present in the chat history, respond with query type as 'context' and output as the reformulated query.

**Note:**
- The reformulated query should always be grammatically correct and contextually rich.
- Only provide the Query Type and Output. Do not provide any other explanation or response.

### Examples

**Example 1:**
Chat History:
user: Who discovered the laws of motion?
ai: Isaac Newton
User Query: Tell me more abt him
Query Type: context
Output: Tell me more about Isaac Newton who discovered the laws of motion.

**Example 2:**
Chat History:

User Query: How are you?
Query Type: general
Output: I am doing well, thank you for asking. How can I help you today?

**Example 3:**
Chat History:
user: Who discovered the laws of motion?
ai: Isaac Newton
User Query: Waht is the capital of France?
Query Type: context
Output: What is the capital of France?

Chat History:
{chat_history}
User Query: {user_query}
"""

# Classify the user query into one of the categories: summary, search, analysis, comparison
query_classification_prompt = """You are an intelligent AI assistant. You will be provided with a user query. Your will have to classify the query into one of the following categories:
- **summary**: The query is either asking for a summary or it requires a large amount of information to be retrieved and summarized.
- **search**: The query is asking for a specific information which can be answered with a single piece of information.
- **analysis**: The query is asking for a thorough analysis of every part of some document or text, which may require reasoning and understanding of the text.
- **comparison**: The query is asking for a comparison between two or more entities, which may require multiple sources of information.

### Examples

**Example 1:**
User Query: In how many cricket world cups, India made it to the top 7?
Answer: summary
**Example 2:**
User Query: Who was the captain of the Indian cricket team in the 2011 world cup?
Answer: search
**Example 3:**
User Query: What are the grammatical and logical errors in this document?
Answer: analysis
**Example 4:**
User Query: What are the differences between the election systems of India and the United States?
Answer: comparison

User Query: {user_query}
Answer:"""

# Classify the user query into one of the categories: summary, search, comparison (when no document is given by the user)
query_classification_prompt_no_doc = """You are an intelligent AI assistant. You will be provided with a user query. Your will have to classify the query into one of the following categories:
- **summary**: The query is either asking for a summary or it requires a large amount of information to be retrieved and summarized.
- **search**: The query is asking for a specific information which can be answered with a single piece of information.
- **comparison**: The query is asking for a comparison between two or more entities, which may require multiple sources of information.

### Examples

**Example 1:**
User Query: In how many cricket world cups, India made it to the top 7?
Answer: summary
**Example 2:**
User Query: Who was the captain of the Indian cricket team in the 2011 world cup?
Answer: search
**Example 4:**
User Query: What are the differences between the election systems of India and the United States?
Answer: comparison

User Query: {user_query}
Answer:"""