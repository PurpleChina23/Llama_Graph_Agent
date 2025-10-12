from langchain_core.tools import tool
from embedding import read_and_query


@tool
def query_knowledge_base(query: str) -> str:
    """Use this tool to query our knowledge base about projects, products, or general info."""
    response = read_and_query(query)
    return str(response)


