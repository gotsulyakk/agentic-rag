from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""

    datasource: Literal["vectorstore", "web_search"] = Field(
        ...,
        description="Route the user query to the vectorstore or websearch. Avalable options are 'vectorstore' or 'web_search'",
    )


llm = ChatOpenAI(temperature=0)
structured_llm_router = llm.with_structured_output(RouteQuery)

message = """You are an expert at routing a user question to a vectorstore or web search.
    The vectorstore contains documents related to machine learning concepts such as: agents, prompt engineering, and adversarial attacks.
    Use the vectorstore for questions on these topics. For ANY other question, choose web-search route."""
router_prompt = ChatPromptTemplate.from_messages(
    [("system", message), ("human", "{question}")]
)

question_router = router_prompt | structured_llm_router
