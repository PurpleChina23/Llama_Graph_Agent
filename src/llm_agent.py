"""
LLM Agent with Tools

This module demonstrates how to create an LLM agent with custom tools using LangChain and LangGraph.
"""

import os
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from typing import Optional
from langgraph.prebuilt import create_react_agent


# ============================================================================
# Load environment variables
# ============================================================================

load_dotenv(find_dotenv())

api_key = os.getenv("OPENAI_API_KEY")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE")

if not api_key:
    print("❌ OPENAI_API_KEY not loaded.")
else:
    print("✅ OPENAI_API_KEY loaded.")

if not OPENAI_API_BASE:
    print("⚠️ OPENAI_API_BASE not found.")
else:
    print(f"✅ OPENAI_API_BASE = {OPENAI_API_BASE}")


# ============================================================================
# Initialize Model
# ============================================================================

llm = init_chat_model(model='gpt-4.1-nano', temperature=0)
print("✅ Model initialized")


# ============================================================================
# Define Tools
# ============================================================================

@tool
def add(a: int, b: int) -> int:
    """Add two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The sum of a and b
    """
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers together.

    Args:
        a: First number
        b: Second number

    Returns:
        The product of a and b
    """
    return a * b


@tool
def search_wikipedia(query: str) -> str:
    """Search Wikipedia for information about a topic.

    Args:
        query: The search query

    Returns:
        A summary of the Wikipedia article
    """
    # This is a mock implementation - you can replace with actual Wikipedia API
    return f"Wikipedia search results for '{query}': This is a mock result. Install wikipedia-api package for real searches."


@tool
def get_weather(location: str, unit: str = "celsius") -> str:
    """Get the current weather for a location.

    Args:
        location: The city or location name
        unit: Temperature unit (celsius or fahrenheit)

    Returns:
        Weather information for the location
    """
    # Mock weather data
    return f"The weather in {location} is currently sunny, 22 degrees {unit}."


# Create a list of all tools
tools = [add, multiply, search_wikipedia, get_weather]
print(f"✅ Defined {len(tools)} tools: {[t.name for t in tools]}")


# ============================================================================
# Bind Tools to LLM
# ============================================================================

llm_with_tools = llm.bind_tools(tools)
print("✅ Tools bound to LLM")


# ============================================================================
# Create Agent with LangGraph
# ============================================================================

agent = create_react_agent(llm, tools)
print("✅ Agent created successfully")


# ============================================================================
# Helper Functions
# ============================================================================

def chat_with_agent(query: str):
    """Chat with the agent and print the conversation.

    Args:
        query: The user's query

    Returns:
        The agent's response
    """
    print(f"\n{'='*60}")
    print(f"User: {query}")
    print(f"{'='*60}")

    response = agent.invoke({"messages": [("user", query)]})

    for message in response["messages"]:
        if message.type == "human":
            print(f"\nUser: {message.content}")
        elif message.type == "ai":
            if message.content:
                print(f"\nAssistant: {message.content}")
            # Check for tool calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    print(f"\n[Tool Call] {tool_call['name']}({tool_call['args']})")
        elif message.type == "tool":
            print(f"[Tool Result] {message.content}")

    print(f"\n{'='*60}\n")
    return response


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Test 1: Simple math calculation
    print("\n" + "="*60)
    print("TEST 1: Simple math calculation")
    print("="*60)
    response = agent.invoke({"messages": [("user", "What is 42 multiplied by 7?")]})
    print("\nAgent response:")
    for message in response["messages"]:
        if message.content:
            print(f"{message.type}: {message.content}")

    # Test 2: Multiple tool usage
    print("\n" + "="*60)
    print("TEST 2: Multiple tool usage")
    print("="*60)
    response = agent.invoke({
        "messages": [("user", "What is 15 plus 28, and then multiply the result by 3?")]
    })
    print("\nAgent response:")
    for message in response["messages"]:
        if message.content:
            print(f"{message.type}: {message.content}")

    # Test 3: Using weather tool
    print("\n" + "="*60)
    print("TEST 3: Using weather tool")
    print("="*60)
    response = agent.invoke({
        "messages": [("user", "What's the weather like in London?")]
    })
    print("\nAgent response:")
    for message in response["messages"]:
        if message.content:
            print(f"{message.type}: {message.content}")

    # Interactive example
    print("\n" + "="*60)
    print("TEST 4: Interactive chat")
    print("="*60)
    chat_with_agent("Calculate 123 multiplied by 456")
