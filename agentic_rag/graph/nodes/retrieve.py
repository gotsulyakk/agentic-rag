from typing import Any, Dict

from graph.state import GraphState
from ingestion import retriever


def retrieve(state: GraphState) -> Dict[str, Any]:
    """
    Retrieve documents from the retriever.

    Args:
        state: The current state of the graph.

    Returns:
        A dictionary containing the retrieved documents and the question
    """
    print("---RETRIEVE---")
    question = state["question"]
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}
