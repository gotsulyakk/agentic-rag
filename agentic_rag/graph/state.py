from typing import List, TypedDict


class GraphState(TypedDict):
    """
    Represents a state of a graph.

    Attributes:
        question: Question
        generation: LLM Generation
        use_web_search: wether to use web search
        documents: List of documents
    """

    question: str
    generation: str
    use_web_search: bool
    documents: List[str]
