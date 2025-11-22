import time
import requests
import rag_evaluation as rag_eval
from rag_evaluation.evaluator import evaluate_response
from test_bed import test_bed

API_URL = "http://localhost:8000/query"

rag_eval.set_api_key("gemini", "AIzaSyDy2JJf-YTeT2F-7tb3B5HvDmpe_6ZmEho")


def query_rag(question: str):
    payload = {"query": question}
    response = requests.post(API_URL, json=payload).json()
    return response["answer"], response["sources"]


def evaluate_two_metrics(query: str, answer: str, document: str):
    """
    Computes only Query Relevance + Faithfulness for a (query, answer, document).
    """
    metric_weights = [0.5, 0.5, 0.0, 0.0, 0.0]

    df = evaluate_response(
        query=query,
        response=answer,
        document=document,
        model_type="gemini",
        model_name="gemini-2.5-flash-preview-09-2025",
        metric_weights=metric_weights
    )

    relevance = float(df.iloc[0]["Score (Normalized)"])
    faithfulness = float(df.iloc[1]["Score (Normalized)"])

    return relevance, faithfulness, df


def run_testbed(test_bed):
    for id, item in enumerate(test_bed):
        if id in [0, 1]: continue
        question = item["question"]
        ideal_answer = item["ideal_answer"]

        print("=" * 80)
        print(f"QUESTION: {question}")
        print(f"IDEAL ANSWER: {ideal_answer}")

        answer, sources = query_rag(question)

        print(f"RAG ANSWER: {answer}")

        relevance, faithfulness, full_df = evaluate_two_metrics(
            question, answer, sources
        )

        print(f"Query Relevance:  {relevance:.3f}")
        print(f"Faithfulness:     {faithfulness:.3f}")

        print("Detailed Evaluator Output:")
        print(full_df)

        full_df.to_csv(f'EVALUATION_RESULT_{id}.csv')

        time.sleep(10)


if __name__ == "__main__":
    run_testbed(test_bed)
