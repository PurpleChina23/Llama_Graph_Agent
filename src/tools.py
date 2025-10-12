"""
Custom tools for the LLM Agent

This module defines various tools that the LLM agent can use to perform tasks.
"""

from langchain_core.tools import tool
from typing import Optional, List, Dict, Any
import requests
import json
from datetime import datetime
import math


@tool
def calculator(operation: str, a: float, b: float) -> float:
    """Perform basic mathematical operations.
    
    Args:
        operation: One of 'add', 'subtract', 'multiply', 'divide', 'power', 'modulo'
        a: First number
        b: Second number
    
    Returns:
        The result of the operation
    """
    try:
        if operation == 'add':
            return a + b
        elif operation == 'subtract':
            return a - b
        elif operation == 'multiply':
            return a * b
        elif operation == 'divide':
            if b == 0:
                return float('inf') if a > 0 else float('-inf')
            return a / b
        elif operation == 'power':
            return a ** b
        elif operation == 'modulo':
            if b == 0:
                return float('nan')
            return a % b
        else:
            return float('nan')
    except Exception as e:
        return float('nan')


@tool
def advanced_calculator(expression: str) -> float:
    """Evaluate a mathematical expression safely.
    
    Args:
        expression: Mathematical expression to evaluate (e.g., "2 + 3 * 4")
    
    Returns:
        The result of the expression
    """
    try:
        # Only allow safe mathematical operations
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return float('nan')
        
        # Use eval with limited globals and locals
        result = eval(expression, {"__builtins__": {}}, {"math": math})
        return float(result)
    except Exception:
        return float('nan')


@tool
def get_current_time() -> str:
    """Get the current date and time.
    
    Returns:
        Current date and time as a string
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_weather_mock(location: str, unit: str = "celsius") -> str:
    """Get mock weather information for a location.
    
    Args:
        location: The city or location name
        unit: Temperature unit (celsius or fahrenheit)
    
    Returns:
        Mock weather information for the location
    """
    # Mock weather data - in a real implementation, you would use a weather API
    import random
    temp = random.randint(15, 30)
    conditions = ["sunny", "cloudy", "rainy", "partly cloudy", "windy"]
    condition = random.choice(conditions)
    
    if unit.lower() == "fahrenheit":
        temp = temp * 9/5 + 32
        unit_str = "°F"
    else:
        unit_str = "°C"
    
    return f"The weather in {location} is currently {condition}, {temp} degrees {unit_str}."


@tool
def search_wikipedia_mock(query: str) -> str:
    """Search Wikipedia for information about a topic (mock implementation).
    
    Args:
        query: The search query
    
    Returns:
        Mock Wikipedia search results
    """
    # Mock Wikipedia search - in a real implementation, you would use the Wikipedia API
    return f"Wikipedia search results for '{query}': This is a mock result. For real Wikipedia searches, install the wikipedia-api package and implement proper API calls."


@tool
def text_analyzer(text: str) -> Dict[str, Any]:
    """Analyze text and return statistics.
    
    Args:
        text: Text to analyze
    
    Returns:
        Dictionary containing text statistics
    """
    words = text.split()
    sentences = text.count('.') + text.count('!') + text.count('?')
    paragraphs = text.count('\n\n') + 1
    
    return {
        "character_count": len(text),
        "word_count": len(words),
        "sentence_count": sentences,
        "paragraph_count": paragraphs,
        "average_word_length": sum(len(word) for word in words) / len(words) if words else 0,
        "unique_words": len(set(word.lower() for word in words))
    }


@tool
def url_checker(url: str) -> Dict[str, Any]:
    """Check if a URL is accessible and return status information.
    
    Args:
        url: URL to check
    
    Returns:
        Dictionary containing URL status information
    """
    try:
        response = requests.get(url, timeout=10)
        return {
            "url": url,
            "status_code": response.status_code,
            "accessible": response.status_code == 200,
            "content_type": response.headers.get('content-type', 'unknown'),
            "content_length": len(response.content)
        }
    except requests.exceptions.RequestException as e:
        return {
            "url": url,
            "status_code": None,
            "accessible": False,
            "error": str(e)
        }


@tool
def json_validator(json_string: str) -> Dict[str, Any]:
    """Validate and parse a JSON string.
    
    Args:
        json_string: JSON string to validate
    
    Returns:
        Dictionary containing validation results
    """
    try:
        parsed = json.loads(json_string)
        return {
            "valid": True,
            "parsed_data": parsed,
            "data_type": type(parsed).__name__
        }
    except json.JSONDecodeError as e:
        return {
            "valid": False,
            "error": str(e),
            "error_position": e.pos
        }


@tool
def list_operations(operation: str, items: List[str], **kwargs) -> List[str]:
    """Perform operations on a list of items.
    
    Args:
        operation: Operation to perform ('sort', 'reverse', 'unique', 'filter')
        items: List of items to operate on
        **kwargs: Additional arguments (e.g., filter_keyword for filter operation)
    
    Returns:
        Resulting list after operation
    """
    try:
        if operation == 'sort':
            return sorted(items)
        elif operation == 'reverse':
            return list(reversed(items))
        elif operation == 'unique':
            return list(dict.fromkeys(items))  # Preserves order
        elif operation == 'filter':
            keyword = kwargs.get('filter_keyword', '')
            return [item for item in items if keyword.lower() in item.lower()]
        else:
            return items
    except Exception:
        return items


@tool
def string_operations(operation: str, text: str, **kwargs) -> str:
    """Perform string operations.
    
    Args:
        operation: Operation to perform ('upper', 'lower', 'title', 'reverse', 'replace')
        text: Input text
        **kwargs: Additional arguments (e.g., old_text, new_text for replace operation)
    
    Returns:
        Resulting string after operation
    """
    try:
        if operation == 'upper':
            return text.upper()
        elif operation == 'lower':
            return text.lower()
        elif operation == 'title':
            return text.title()
        elif operation == 'reverse':
            return text[::-1]
        elif operation == 'replace':
            old_text = kwargs.get('old_text', '')
            new_text = kwargs.get('new_text', '')
            return text.replace(old_text, new_text)
        else:
            return text
    except Exception:
        return text


# List of all available tools
ALL_TOOLS = [
    calculator,
    advanced_calculator,
    get_current_time,
    get_weather_mock,
    search_wikipedia_mock,
    text_analyzer,
    url_checker,
    json_validator,
    list_operations,
    string_operations
]


def get_tools() -> List:
    """Get all available tools.
    
    Returns:
        List of all tool functions
    """
    return ALL_TOOLS


def get_tool_names() -> List[str]:
    """Get names of all available tools.
    
    Returns:
        List of tool names
    """
    return [tool.name for tool in ALL_TOOLS]


if __name__ == "__main__":
    # Test the tools
    print("Available tools:")
    for tool in ALL_TOOLS:
        print(f"- {tool.name}: {tool.description}")
    
    # Test calculator
    result = calculator.invoke({"operation": "multiply", "a": 5, "b": 7})
    print(f"\nTest: 5 * 7 = {result}")
    
    # Test text analyzer
    text_result = text_analyzer.invoke({"text": "Hello world! This is a test."})
    print(f"\nText analysis: {text_result}")