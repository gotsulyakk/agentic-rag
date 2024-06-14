from dotenv import load_dotenv

from graph.graph import app


load_dotenv()


if __name__ == "__main__":
    question = "What is agent memory in context of LLMs?"
    print(app.invoke(input={"question": question}))
