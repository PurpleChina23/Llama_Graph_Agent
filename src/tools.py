from langchain_core.tools import tool
from embedding import read_and_query, read_and_retrieve

# ============================================================================
# IMPROVEMENT NEEDED: Add better imports for error handling
# ============================================================================
# TODO: Add these imports:
# import logging
# from typing import Optional
#
# logger = logging.getLogger(__name__)
# ============================================================================


@tool
def query_knowledge_base(query: str) -> str:
    # ============================================================================
    # IMPROVEMENT NEEDED: Enhance docstring (2025 best practice)
    # ============================================================================
    # Research shows: Well-documented tools improve LLM tool selection by 30-40%
    #
    # Current docstring is too vague: "projects, products, or general info"
    # LLM doesn't know when to use this vs other tools
    #
    # TODO: Expand docstring to include:
    # 1. Detailed description of what the tool does
    # 2. Specific use cases (when TO use it)
    # 3. Non-use cases (when NOT to use it)
    # 4. Parameter descriptions with examples
    # 5. Return value description
    # 6. Example usage
    #
    # See IMPLEMENTATION_PLAN.md Section 8 for complete example docstring
    # ============================================================================
    """Use this tool to query our knowledge base about projects, products, or general info."""
    # TODO: Replace with detailed docstring (see IMPLEMENTATION_PLAN.md)

    # ============================================================================
    # IMPROVEMENT NEEDED: Add error handling
    # ============================================================================
    # TODO: Wrap in try-except block:
    # try:
    #     response = read_and_query(query)
    #     if not response:
    #         return "No information found in knowledge base."
    #     return str(response)
    # except Exception as e:
    #     logger.error(f"Error in query_knowledge_base: {e}")
    #     return f"Error querying knowledge base: {str(e)}"
    # ============================================================================

    response = read_and_query(query)
    return str(response)

# ============================================================================
# 🔴 CRITICAL BUG: Incomplete function!
# ============================================================================
# This function has multiple issues:
# 1. Missing @tool decorator (won't be available to the agent)
# 2. Typo in name: "retrieved" should be "retrieve"
# 3. Missing docstring
# 4. No error handling
# 5. Incorrect return format (should format nodes nicely)
#
# PRIORITY: CRITICAL - Fix before using
# ============================================================================
def retrieved_knowledge_base(query: str) -> str:  # 🔴 TYPO: Should be "retrieve_knowledge_base"
    # 🔴 MISSING: @tool decorator above this function

    # ============================================================================
    # IMPROVEMENT NEEDED: Add comprehensive docstring
    # ============================================================================
    # TODO: Add detailed docstring explaining:
    # - This retrieves raw chunks without LLM synthesis (faster than query_knowledge_base)
    # - When to use: Need exact specs, fast retrieval, multiple perspectives
    # - Args and return value descriptions
    # - Examples
    # ============================================================================

    # ============================================================================
    # IMPROVEMENT NEEDED: Add error handling
    # ============================================================================
    # TODO: Wrap in try-except
    # TODO: Check if nodes is empty
    # ============================================================================

    # ============================================================================
    # IMPROVEMENT NEEDED: Format output better
    # ============================================================================
    # TODO: Format nodes with clear separation:
    # if not nodes:
    #     return "No relevant information found."
    # context_str = "\n\n--- Document Chunk ---\n\n".join([node.get_content() for node in nodes])
    # return context_str
    # ============================================================================

    response = read_and_retrieve(query)  # Returns list of nodes, not str
    return str(response)  # This won't format well - returns repr(nodes)


# ============================================================================
# CORRECTED VERSION (TODO: Replace above function with this)
# ============================================================================
# @tool
# def retrieve_knowledge_base(query: str) -> str:
#     """
#     Retrieve raw document chunks from the golf equipment knowledge base.
#
#     This tool retrieves relevant text chunks directly from documents without
#     LLM synthesis. It's faster than query_knowledge_base and returns exact
#     text from source documents.
#
#     Use this tool when:
#     - You need exact specifications or quotes from documents
#     - You want multiple perspectives on a topic
#     - Speed is important (no LLM synthesis overhead)
#     - You want to see all relevant information before synthesizing
#
#     Do NOT use when:
#     - User wants a direct answer (use query_knowledge_base instead)
#     - You need the LLM to interpret or summarize information
#
#     Args:
#         query: Search query to find relevant document chunks.
#                Examples: "driver loft specifications", "TaylorMade Qi35",
#                         "shaft flex for high swing speed"
#
#     Returns:
#         Formatted text chunks from the most relevant documents, separated
#         by "--- Document Chunk ---". Returns error message if retrieval fails.
#
#     Example:
#         >>> retrieve_knowledge_base("TaylorMade Qi35 specifications")
#         "--- Document Chunk ---\n\nThe TaylorMade Qi35 driver features..."
#     """
#     try:
#         nodes = read_and_retrieve(query)
#         if not nodes:
#             return "No relevant information found in the knowledge base for this query."
#
#         # Format nodes with clear separation
#         context_str = "\n\n--- Document Chunk ---\n\n".join([
#             node.get_content() for node in nodes
#         ])
#         return context_str
#     except Exception as e:
#         return f"Error retrieving from knowledge base: {str(e)}"
# ============================================================================


