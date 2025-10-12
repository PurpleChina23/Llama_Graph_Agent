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


class LLMAgent:
    """LLM Agent with tools and memory capabilities."""
    
    def __init__(self, model_name: str = 'gpt-4o-mini', temperature: float = 0):
        """Initialize the LLM Agent.
        
        Args:
            model_name: The model to use
            temperature: Temperature for the model
        """
        self.model_name = model_name
        self.temperature = temperature
        self.llm = None
        self.agent = None
        self.tools = self._get_tools()
        self._initialize()
    
    def _initialize(self):
        """Initialize the agent components."""
        self._load_environment()
        self._initialize_model()
        self._create_agent()
    
    def _load_environment(self):
        """Load environment variables."""
        # Try to load from current directory first, then from src directory
        env_path = find_dotenv()
        if not env_path:
            # If not found, try the src directory
            src_env_path = os.path.join(os.path.dirname(__file__), '.env')
            if os.path.exists(src_env_path):
                load_dotenv(src_env_path)
            else:
                load_dotenv()
        else:
            load_dotenv(env_path)
        
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.api_base = os.getenv("OPENAI_API_BASE")
        
        if not self.api_key:
            print("❌ OPENAI_API_KEY not loaded.")
        else:
            print("✅ OPENAI_API_KEY loaded.")
        
        if not self.api_base:
            print("⚠️ OPENAI_API_BASE not found.")
        else:
            print(f"✅ OPENAI_API_BASE = {self.api_base}")
    
    def _initialize_model(self):
        """Initialize the language model."""
        if not self.api_key:
            print("❌ Cannot initialize model without API key")
            return
        
        try:
            self.llm = init_chat_model(
                model=self.model_name, 
                temperature=self.temperature
            )
            print(f"✅ Model initialized: {self.model_name}")
        except Exception as e:
            print(f"❌ Failed to initialize model: {e}")
            self.llm = None
    
    def _create_agent(self):
        """Create the agent with tools."""
        if not self.llm:
            print("❌ Cannot create agent without initialized model")
            return
        
        try:
            self.agent = create_react_agent(self.llm, self.tools)
            print("✅ Agent created successfully")
        except Exception as e:
            print(f"❌ Failed to create agent: {e}")
            self.agent = None
    
    def _get_tools(self):
        """Get the list of available tools."""
        return [add, multiply, search_wikipedia, get_weather]
    
    def chat(self, query: str) -> dict:
        """Chat with the agent.
        
        Args:
            query: The user's query
            
        Returns:
            The agent's response
        """
        if not self.agent:
            return {"error": "Agent not initialized"}
        
        try:
            response = self.agent.invoke({"messages": [("user", query)]})
            return response
        except Exception as e:
            return {"error": f"Failed to get response: {e}"}
    
    def chat_with_display(self, query: str) -> dict:
        """Chat with the agent and display the conversation.
        
        Args:
            query: The user's query
            
        Returns:
            The agent's response
        """
        print(f"\n{'='*60}")
        print(f"User: {query}")
        print(f"{'='*60}")
        
        response = self.chat(query)
        
        if "error" in response:
            print(f"❌ Error: {response['error']}")
            return response
        
        # Display the conversation
        if "messages" in response:
            for message in response["messages"]:
                if hasattr(message, 'type'):
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
    
    def is_ready(self) -> bool:
        """Check if the agent is ready to use.
        
        Returns:
            True if agent is ready, False otherwise
        """
        return self.agent is not None and self.llm is not None


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


# ============================================================================
# Factory Functions
# ============================================================================

def create_agent(model_name: str = 'gpt-4o-mini', temperature: float = 0) -> LLMAgent:
    """Create a new LLM agent instance.
    
    Args:
        model_name: The model to use
        temperature: Temperature for the model
        
    Returns:
        A new LLMAgent instance
    """
    return LLMAgent(model_name=model_name, temperature=temperature)


# ============================================================================
# Main Execution
# ============================================================================

if __name__ == "__main__":
    # Create agent instance
    agent = create_agent()
    
    if not agent.is_ready():
        print("❌ Agent failed to initialize. Please check your API key and try again.")
        exit(1)
    
    print(f"✅ Agent ready with {len(agent.tools)} tools")
    
    # Test 1: Simple math calculation
    print("\n" + "="*60)
    print("TEST 1: Simple math calculation")
    print("="*60)
    response = agent.chat_with_display("What is 42 multiplied by 7?")

    # Test 2: Multiple tool usage
    print("\n" + "="*60)
    print("TEST 2: Multiple tool usage")
    print("="*60)
    response = agent.chat_with_display("What is 15 plus 28, and then multiply the result by 3?")

    # Test 3: Using weather tool
    print("\n" + "="*60)
    print("TEST 3: Using weather tool")
    print("="*60)
    response = agent.chat_with_display("What's the weather like in London?")