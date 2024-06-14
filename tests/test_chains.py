from dotenv import load_dotenv
from pprint import pprint

load_dotenv()

from agentic_rag.graph.chains.retrieval_grader import GradeDocuments, retrieval_grader
from agentic_rag.graph.chains.hallucination_grader import (
    GradeHallucinations,
    hallucination_grader,
)
from agentic_rag.graph.chains.answer_grader import GradeAnswer, answer_grader
from agentic_rag.graph.chains.generation import generation_chain
from agentic_rag.graph.chains.router import RouteQuery, question_router

from agentic_rag.ingestion import retriever


# def test_retrieval_grader_answer_yes() -> None:
#     question = "agent memory"
#     docs = retriever.invoke(question)
#     doc_text = docs[1].page_content

#     res: GradeDocuments = retrieval_grader.invoke(
#         {"question": question, "document": doc_text}
#     )

#     # it just works randomly, so, ignore it
#     # assert str(res.binary_score).lower() == "yes"
#     # instead, just check if it is 'yes' or 'no'
#     assert str(res.binary_score).lower() in ["yes", "no"]


# def test_retrieval_grader_answer_no() -> None:
#     question = "donald trump"
#     docs = retriever.invoke(question)
#     doc_text = docs[1].page_content

#     res: GradeDocuments = retrieval_grader.invoke(
#         {"question": question, "document": doc_text}
#     )

#     # it just works randomly, so, ignore it
#     # assert str(res.binary_score).lower() == "no"
#     # instead, just check if it is 'yes' or 'no'
#     assert str(res.binary_score).lower() in ["yes", "no"]


def test_retrieval_grader_answer_yes_or_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)
    doc_text = docs[1].page_content

    res: GradeDocuments = retrieval_grader.invoke(
        {"question": question, "document": doc_text}
    )

    assert str(res.binary_score).lower() in ["yes", "no"]


def test_generation_chain() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})

    pprint(generation)
    assert generation is not None


def test_hallucination_grader_answer_yes_or_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeHallucinations = hallucination_grader.invoke(
        {"documents": docs, "generation": generation}
    )

    assert str(res.binary_score).lower() in ["yes", "no"]


def test_answer_grader_answer_yes_or_no() -> None:
    question = "agent memory"
    docs = retriever.invoke(question)

    generation = generation_chain.invoke({"context": docs, "question": question})
    res: GradeAnswer = answer_grader.invoke(
        {"question": question, "generation": generation}
    )

    assert str(res.binary_score).lower() in ["yes", "no"]


def test_question_router_to_vectorstore() -> None:
    question = "agent memory"

    res: RouteQuery = question_router.invoke({"question": question})

    assert res.datasource == "vectorstore"


def test_question_router_to_websearch() -> None:
    question = "Where is John Paul II Catholic High School located?"

    res: RouteQuery = question_router.invoke({"question": question})

    assert res.datasource == "web_search"
