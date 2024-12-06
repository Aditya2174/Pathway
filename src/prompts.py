unsafe_query_resp = "I am sorry, I cannot process this query. If you have any other queries, please feel free to ask."

evaluate_sufficiency_prompt = """You are an intelligent AI assistant. You will be provided with a user query and a context. Your task is to evaluate whether the context provided is sufficient to answer the user query.
The context may not be enough to fully answer the query, but provided it is relevant, you should respond with 'yes'. In any other case, respond with 'no'.
Only respond with 'yes' or 'no'. Do not provide any other explanation or response.

Query: {query}
Context:
{context}
Response:"""

# Reformulate the query and find the type of the query
reformulation_type_prompt = """You are an intelligent AI assistant. The current date in YYYY-MM-DD format is {current_date}. You will be provided with:
1. Chat history between the user and the chatbot.
2. A user-uploaded document, which might be a summary of a larger document rather than the complete document.
3. A user query.

Your task is to determine the query type and generate an output according to the following rules:
- If the query is chit-chat or conversational in nature (e.g., greetings like 'hello', 'hi', or expressions of gratitude like 'thanks') and doesn't require any particular information to be answered, respond with query type as 'general' and output as a polite response to the query.
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
- If the query is chit-chat or conversational in nature (e.g., greetings like 'hello', 'hi', or expressions of gratitude like 'thanks') and doesn't require any particular information to be answered, respond with query type as 'general' and output as a polite response to the query.
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

# Classify the user query into one of these categories: code_execution, summary, search, analysis, comparison
query_classification_prompt = """You are an intelligent AI assistant. You will be provided with a user query. You will have to classify the query into one of the following categories:
- **code_execution**: The query specifically requires a code to be **executed**, such as plotting, performing numerical calculations, or verifying the output of a code. **Note:** If the user only asks to write the code and not execute it, do not classify it as code_execution.
- **summary**: The query is either asking for a summary or it requires a large amount of information to be retrieved and summarized.
- **search**: The query is asking for a specific information which can be answered with a single piece of information.
- **analysis**: The query is asking for a thorough analysis of every part of some document or text, which may require reasoning and understanding of the text.
- **comparison**: The query is asking for a comparison between two or more entities, which may require multiple sources of information.

**Note:**
- If you are not confident about the classification, respond with 'other'.
- Only provide your answer in lowercase. Do not provide any other explanation or response.

### Examples

**Example 1:**
User Query: Plot a sine wave.
Answer: code_execution
**Example 2:**
User Query: In how many cricket world cups, India made it to the top 7?
Answer: summary
**Example 3:**
User Query: Who was the captain of the Indian cricket team in the 2011 world cup?
Answer: search
**Example 4:**
User Query: What are the grammatical and logical errors in this document?
Answer: analysis
**Example 5:**
User Query: What are the differences between the election systems of India and the United States?
Answer: comparison
**Example 6:**
User Query: Write a code to plot a sine wave.
Answer: search

User Query: {user_query}
Answer:"""

# Don't classify into analysis if document is not given
query_classification_prompt_no_doc = """You are an intelligent AI assistant. You will be provided with a user query. You will have to classify the query into one of the following categories:
- **code_execution**: The query specifically requires a code to be **executed**, such as plotting, performing numerical calculations, or verifying the output of a code. **Note:** If the user only asks to write the code and not execute it, do not classify it as code_execution.
- **summary**: The query is either asking for a summary or it requires a large amount of information to be retrieved and summarized.
- **search**: The query is asking for a specific information which can be answered with a single piece of information.
- **comparison**: The query is asking for a comparison between two or more entities, which may require multiple sources of information.

**Note:**
- If you are not confident about the classification, respond with 'other'.
- Only provide your answer in lowercase. Do not provide any other explanation or response.

### Examples

**Example 1:**
User Query: Plot a sine wave.
Answer: code_execution
**Example 2:**
User Query: In how many cricket world cups, India made it to the top 7?
Answer: summary
**Example 3:**
User Query: Who was the captain of the Indian cricket team in the 2011 world cup?
Answer: search
**Example 4:**
User Query: What are the differences between the election systems of India and the United States?
Answer: comparison
**Example 5:**
User Query: Write a code to plot a sine wave.
Answer: search

User Query: {user_query}
Answer:"""

agent_system_prompt = """You are a helpful AI assistant with expertise in legal and financial matters and general knowledge of the world. The current date in YYYY-MM-DD format is {current_date}. Additionally, you have the ability to write code when required to assist with queries. You will be provided with a query and context relevant to the query.

Your task is to:
1. Provide a concise and informative response to the query, using only the context provided.
2. If you are not confident about the answer based on the given context, politely state that you are not sure about the answer.
3. Avoid phrases like "based on the context provided" or "based on the documents" in your response; just directly answer the query.
4. If the context contains contradictory or non-entailing pieces of information relevant to the query:
   - Summarize the response incorporating all the relevant information.
   - Ask the user if they would like an answer based on any specific piece of information.
5. If the query is vague (e.g., "Tell me about Python syntax"):
   - Provide a summary response based on the entire context.
   - Politely ask the user if they can be more specific about what they are looking for.
6. If the query involves coding, generate accurate and efficient code solutions and nothing else. Ensure your code is well-commented and tailored to the user's request. **Note:** if the query requires generating a plot, do not use interactive commands like `plt.show`. Instead, save the plot to the path `plots/image_{plot_count}.png` only.

**Note:**
- If the query is related to code writing and the provided context is not relevant to the query, proceed with writing the code and ignore the context.

Be clear, accurate, and avoid unnecessary elaboration."""

user_proxy_prompt = """If you have more information to share about the same, state it; otherwise, strictly respond with 'done' and nothing else."""

hyde_prompt = """If you have knowledge about the topic write a passage that answers the given query, else if you are not aware of the answer expand the given query into a text document for retrieval adding similar keywords and avoid ambiguous words or questions.Return only 3-4 sentences.
Query: what does chattel mean on credit history
Passage: A “Chattel” notation on a credit history report is a type of loan that is secured by a person’s tangible property. This type of loan typically includes items such as vehicles, furniture and appliances. Chattel loans are typically obtained for the purposes of making major purchases, such as a car or home appliances. By using the credit report, lenders can check to see if a person has taken out a chattel loan in the past.
Query: How much does Microsoft earn annually?
Passage: Microsoft's annual earnings.  Information sought pertains to Microsoft's total annual revenue or profit.  The query aims to retrieve data on the company's financial performance on a yearly basis.
Query: {input_query}
Passage: """