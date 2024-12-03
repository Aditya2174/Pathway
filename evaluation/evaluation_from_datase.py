from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.evaluation import evaluate, LangChainStringEvaluator
LANGCHAIN_API_KEY= "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"

LLM_API_KEY= "AIzaSyCDfqPNXPPfBsW5fgGbmiMfro5loK15wt0"
import os

os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY 
os.environ["LANGCHAIN_TRACING_V2"] = "true" 
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
os.environ["GOOGLE_API_KEY"] = LLM_API_KEY

from langchain_google_genai import ChatGoogleGenerativeAI

# Update system message if desired.
system_message = "You are a chatbot."

# Target task definition.
# prompt = ChatPromptTemplate.from_messages([
#   ("system", system_message),
#   ("user", "{input}")
# ])
# chat_model = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
output_parser = StrOutputParser()

# chain = prompt | chat_model | output_parser

_PROMPT_TEMPLATE = """You are an expert professor specialized in grading students' answers to questions.
You are grading the following question:
{query}
Here is the real answer:
{answer}
You are grading the following predicted answer:
{result}
Response with a valid grade from 1 - 10. Also, give valid reasons for the same.
Response should be like:
Reason: ...
Score : ...
"""
from langchain_core.prompts.prompt import PromptTemplate
PROMPT = PromptTemplate(
  input_variables=["query", "answer", "result"], template=_PROMPT_TEMPLATE
)
eval_llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
# The name or UUID of the LangSmith dataset to evaluate on.
# Alternatively, you can pass an iterator of examples
data = "Correct Pipeline"

# A string to prefix the experiment name with.
# If not provided, a random string will be generated.
experiment_prefix = "Anather One"

# List of evaluators to score the outputs of target task 


  
  
def prepare_data(run, example):
  return {
          "input" : example.inputs['question'],
          "prediction" : example.inputs['response'], 
          "reference": example.inputs['ground_truth'],
          "retrieved_context": ('No context retrieved' if example.inputs['retrieved_contexts'] == ''  else example.inputs['retrieved_contexts'])
  }
  
import regex


def faithfulness_score(run, example) ->float:
    prompt_template ="""[Instruction]  
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant based on **faithfulness** to the retrieved context provided below.  
Your task is to assess whether the AI’s response aligns accurately with the information found in the retrieved context. You must consider if the response is consistent with the context, free of contradictions, and relevant to the information retrieved.

[Retrieved Contexts]  
{retrieved_contexts}  
Begin your evaluation by providing a brief explanation. Be objective and precise in your reasoning. After providing your explanation, rate the response on a scale from 1 to 10, strictly following this format: "[[rating]]", for example: "Rating: [[5]]".

[The Start of Assistant's Answer]  
{prediction}  
[The End of Assistant's Answer]
"""

    
    faithful_prompt = PromptTemplate(
        input_variables=["retrieved_contexts", "prediction"], template=prompt_template)
    
    output = eval_llm.invoke(faithful_prompt.invoke({
      "retrieved_contexts": example.inputs['retrieved_contexts'],
      "prediction": example.inputs['response']
      })).content
    
    score = regex.search(r"\[\[(\d+(\.\d*)?)\]\]", output).group(1)
    
    return {
      "score": float(score)/10,
      "reason": output,
    }
    
    
    
    
    
    

  
try:
  evaluators = [
    # LangChainStringEvaluator("cot_qa", config={"llm" : eval_llm}, prepare_data= lambda run, example: {"prediction" : example.outputs['response'], "reference" : example.outputs['ground_truth'], "input" : example.inputs['user_input']}),
    # LangChainStringEvaluator("labeled_criteria", config={"criteria": "depth", "llm" : eval_llm},prepare_data= lambda run, example: {"prediction" : example.outputs['response'], "reference" : example.outputs['ground_truth'], "input" : example.inputs['user_input']}),
    LangChainStringEvaluator("labeled_score_string", config={ "criteria": "helpfulness", "normalize_by": 10, "llm" : eval_llm },prepare_data=prepare_data), # relevance
    
    LangChainStringEvaluator("labeled_score_string", config={ "criteria": "correctness", "normalize_by": 10, "llm" : eval_llm },prepare_data=prepare_data), # 
    faithfulness_score
  ]

  # Custom evaluators information can be found here: https://docs.smith.langchain.com/evaluation/how_to_guides/evaluation/evaluate_llm_application#use-custom-evaluators

  # Evaluate the target task
  results = evaluate(
    lambda input: input,
    data=data,
    evaluators=evaluators,
    experiment_prefix=experiment_prefix,
    max_concurrency=1
  )
except Exception as e:
  print(e)
  exit(0)