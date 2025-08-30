# --- LangSmith Evaluation Setup ---
import os
from dotenv import load_dotenv

from langsmith import Client

# --- Load env
load_dotenv()

def get_langsmith_client():
    api_key = os.getenv("LANGSMITH_API_KEY")
    if not api_key:
        raise RuntimeError("Missing LANGSMITH_API_KEY. Add it to .env or Streamlit secrets.")
    return Client(api_key=api_key)


# Set tracing env vars
os.environ["LANGSMITH_TRACING"] = "true"


# --- Log Dataset & Example in LangSmith ---
client = get_langsmith_client()
dataset_name= "sample_dataset"

# *****************************************************************************************************************
# Create examples
examples = [
    {
        "inputs": {"question": "data\gaint_receipt1.png"},
        "outputs": {
            "answer": "The Giant store located at 2425 Centerville Road, Herndon, VA 20171 has a total of 13 items in its inventory."
                      " The total cost of these items is $39.30. The store offers various products including bread, vegetables, and household items."},
    },
    {
        "inputs": {"question": "data\\braums_bill.jpeg"},
        "outputs": {
            "answer": "A customer purchased 2 items from Braum's store located at 1222 W MCDERMOTT DR, ALEN, TX 75013."
                      " The items included 2 GAL Milk and 1 YG Capp Chunk Choc. The total cost of the purchase was $13.97."},
    },
]

try:
    dataset = client.create_dataset(
        dataset_name=dataset_name, description="Dataset for receipt extractor."
    )

    # Add the examples to the dataset
    client.create_examples(dataset_id=dataset.id, examples=examples)

except Exception as e:
    print("Dataset already exists, using the existing dataset.")
    # --------------Target Function for Evaluation----------------

from src.agents.receipt_extractor.agent_swetha import target_eval, llm
def target(inputs: dict) -> dict:
    return target_eval(inputs)

## Define evaluator
from openevals.llm import create_llm_as_judge
CORRECTNESS_PROMPT = """
You are an evaluator. Compare the model output with the reference output.
See if they match semantically.

Return JSON ONLY in the following format:
{{
    "reasoning": "<short explanation>",
    "score": true or false
}}

Example:
{{
    "reasoning": "Both outputs match semantically.", 
    "score": true
}}
"""
from pydantic import BaseModel
from langsmith.evaluation.evaluator import EvaluationResult
class ScoreSchema(BaseModel):
    reasoning: str
    score: bool

def correctness_evaluator(inputs: dict, outputs: dict, reference_outputs: dict):
    evaluator = create_llm_as_judge(
        prompt=CORRECTNESS_PROMPT,
        judge=llm,
        output_schema=ScoreSchema
    )
    eval_result = evaluator(
    inputs=inputs,
    outputs=outputs,
    reference_outputs=reference_outputs
    )
    result = eval_result.__dict__
    reasoning = result.get("reasoning", "")
    score = result.get("score", False)

    return EvaluationResult(
         key="correctness",
         score=score,
         comment=reasoning
    )

## Run and View the results

# After running the evaluation, a link will be provided to view the results in langsmith
experiment_results = client.evaluate(
    target,
    data=dataset_name,
    evaluators=[
        correctness_evaluator,
    ],
    experiment_prefix="first-eval-in-langsmith",
    max_concurrency=2,
)