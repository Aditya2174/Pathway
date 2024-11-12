import os
import pandas as pd
from ragas.metrics import LLMContextRecall, Faithfulness, AnswerRelevancy
from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from ragas.evaluation import Dataset
import ast
from time import sleep
from ragas import EvaluationDataset

class QAEvaluator:
    def __init__(self, df, api_key, model="gpt-4o mini"):
        """
        Initializes the QAEvaluator with a pandas DataFrame, API key, and model type.
        
        Parameters:
        - df (pd.DataFrame): The dataset in pandas DataFrame format.
        - api_key (str): The OpenAI API key for LLM access.
        - model (str): The LLM model name (default: "gpt-4o").
        """
        self.api_key = api_key
        self.model = model
        os.environ["OPENAI_API_KEY"] = self.api_key
        self.df = df
        self.evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model=self.model))
        self.metrics = self._initialize_metrics()

    def _initialize_metrics(self):
        """
        Initializes and returns a list of metrics to evaluate the dataset.
        
        Returns:
        - metrics (list): List of metric instances.
        """
        return [
            LLMContextRecall(llm=self.evaluator_llm),  # Requires ground truth and context
            Faithfulness(llm=self.evaluator_llm),       # Requires question, context, and answer
            AnswerRelevancy(llm=self.evaluator_llm)     # Requires question and answer
        ]

    def evaluate(self):
        """
        Evaluates the dataset row-by-row using the initialized metrics.
        
        Returns:
        - pd.DataFrame: DataFrame containing the evaluation results.
        """
        # List to store results for each row
        results_list = []
        df.rename(columns={"response" : "answer"})
        df['retrieved_contexts'] = df['retrieved_contexts'].apply(ast.literal_eval)
        for i, row in df.iterrows():
            cur_df = pd.DataFrame([row])
            print(f"Evaluating row {i}:\n{cur_df}")
            input_dataset = EvaluationDataset.from_pandas(cur_df)
            result = evaluate(
                dataset=input_dataset,
                metrics=self.metrics
            )
            sleep(65)

            results_list.append(result.to_pandas().to_dict(orient="records")[0])

        print(f"Evaluated successfully.")
        
        # Convert the list of results to a DataFrame for easy viewing
        results_df = pd.DataFrame(results_list)
        return results_df

# Usage example:
# Assuming 'df' is your input pandas DataFrame containing the necessary columns.
api_key = "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"
df = pd.read_csv("final.csv")
evaluator = QAEvaluator(df=df, api_key=api_key)
results_df = evaluator.evaluate()
results_df.to_csv("result.csv", index=False)

