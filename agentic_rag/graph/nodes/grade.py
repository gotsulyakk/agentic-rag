from typing import Any, Dict

from graph.chains.retrieval_grader import retrieval_grader
from graph.state import GraphState


def grade_documents(state: GraphState) -> Dict[str, Any]:
    """
    Determines whether the retrieved documents are relevant to the user question.
    If any document is not relevant, we will set a flag to run web search.

    Args:
        state (dict): The current state of the graph.

    Returns:
        state (dict): Filtered out irrelevant documents and updated use_web_search state.
    """
    print("---GRADE DOCUMENTS---")
    question = state["question"]
    documents = state["documents"]

    filtered_documents = []
    use_web_search = False
    for doc in documents:
        result = retrieval_grader.invoke(
            {"question": question, "document": doc.page_content}
        )
        grade = result.binary_score
        if grade.lower() == "yes":
            print("---DOCUMENT IS RELEVANT---")
            filtered_documents.append(doc)
        else:
            print("---DOCUMENT IS NOT RELEVANT---")
            use_web_search = True
            continue
    return {
        "documents": filtered_documents,
        "use_web_search": use_web_search,
        "question": question,
    }
