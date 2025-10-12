{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "1",
   "metadata": {},
   "source": [
    "# Tools for LLM Agent\\n",
    "\\n",
    "This notebook demonstrates how to create custom tools for your LLM agent."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "2",
   "metadata": {},
   "outputs": [],
   "source": [
    "from langchain_core.tools import tool\\n",
    "from typing import Optional\\n",
    "\\n",
    "print(\\\"✅ Imports successful\\\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3",
   "metadata": {},
   "source": [
    "## Example Tools"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4",
   "metadata": {},
   "outputs": [],
   "source": [
    "@tool\\n",
    "def calculator(operation: str, a: float, b: float) -> float:\\n",
    "    \\\"\\\"\\\"Perform basic mathematical operations.\\n",
    "    \\n",
    "    Args:\\n",
    "        operation: One of 'add', 'subtract', 'multiply', 'divide'\\n",
    "        a: First number\\n",
    "        b: Second number\\n",
    "    \\n",
    "    Returns:\\n",
    "        The result of the operation\\n",
    "    \\\"\\\"\\\"\\n",
    "    if operation == 'add':\\n",
    "        return a + b\\n",
    "    elif operation == 'subtract':\\n",
    "        return a - b\\n",
    "    elif operation == 'multiply':\\n",
    "        return a * b\\n",
    "    elif operation == 'divide':\\n",
    "        if b == 0:\\n",
    "            return \\\"Error: Division by zero\\\"\\n",
    "        return a / b\\n",
    "    else:\\n",
    "        return \\\"Error: Invalid operation\\\"\\n",
    "\\n",
    "print(\\\"✅ Calculator tool defined\\\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test the calculator tool\\n",
    "result = calculator.invoke({\\\"operation\\\": \\\"multiply\\\", \\\"a\\\": 5, \\\"b\\\": 7})\\n",
    "print(f\\\"5 * 7 = {result}\\\")"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
