import os
from pathway.xpacks.llm.vector_store import VectorStoreClient 
from pathway.xpacks.llm.llms import LiteLLMChat, prompt_chat_single_qa
from custom_llm_chat import CustomLiteLLMChat
import os
import pathway as pw
# set env
os.environ["GEMINI_API_KEY"] = "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc" # use your api key here

VECTOR_STORE_HOST = "127.0.0.1"
VECTOR_STORE_PORT = 8755
FIXED_QUERY = input("Enter your query: ")
LLM_API_KEY = "AIzaSyB_ic4AmBCWeFGnhV4WcVyU9GKPRRVQTyc"

import logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

client = VectorStoreClient(host=VECTOR_STORE_HOST, port=VECTOR_STORE_PORT, timeout=1000)


model = CustomLiteLLMChat(
    model="gemini/gemini-1.5-flash",
    api_key=LLM_API_KEY,
    temperature=0.0,
    max_tokens=100,
)


documents = client(FIXED_QUERY, k= 10)
print(f"Retrieved {len(documents)} documents.")
# print(documents)

context = "\n\n".join(doc.get('text', '') for doc in documents)


prompt = f"Context:\n{context}\n\nQuestion: {FIXED_QUERY}\n"
# res = prompt_chat_single_qa(prompt)
# print(f'\n\nResponse: {res}')
messages = [
    {
        "role": "user",
        "content": (
            prompt
        )
    }
]

print(context)
# Call the UDF correctly
# response =  model(prompt)  # Using a helper method for direct invocation

print("\n--- LLM Response ---")
# print(response)
# pw.run()
response = model.get_response(messages)
print(f'\n\nResponse: {response}')
# messages = [{"role": "user", "content": prompt}]

# response = completion(
#     model="gemini/gemini-1.5-flash",
#     messages=messages,
# )

# print("\n--- LLM Response ---")
# print(response['choices'][0]['message']['content'])