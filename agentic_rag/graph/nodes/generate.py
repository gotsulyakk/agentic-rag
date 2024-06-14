from typing import Any, Dict

from graph.state import GraphState
from graph.chains.generation import generation_chain


def generate(state: GraphState) -> Dict[str, Any]:
    """
    Generate a response to the user question.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): A dictionary containing the generated response and the question
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    generation = generation_chain.invoke({"context": documents, "question": question})
    return {"generation": generation, "documents": documents, "question": question}
