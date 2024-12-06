import requests
from typing import Annotated
import streamlit as st

google_api_key = "AIzaSyD939q3PbECaSJO1IAzRbmpqlREgJteLKg"
google_cse_id = "d3a75edbda16e451e"
tool_usage_log = {} # Dictionary to track tool usage

# Function to perform web search using Google Custom Search API
def web_search(query: str, api_key: str, cse_id: str, num_results: int = 5) -> str:
    """
    Perform a web search using Google Custom Search API.
    :param query: The search query string.
    :param api_key: Google API key.
    :param cse_id: Google Custom Search Engine ID.
    :param num_results: Number of search results to retrieve (default: 5).
    :return: A formatted string of search results or an error message.
    """
    url = "https://www.googleapis.com/customsearch/v1"
    # url = "https://cse.google.com/cse"
    params = {
        "key": api_key,
        "cx": cse_id,
        "q": query,
        "num": num_results
    }
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        print(f"Search response: {response}")
        print(f"Response content: {response.content}")
        data = response.json()
        items = data.get('items', [])
        if items:
            return "\n".join([f"{item['title']} - {item['link']}" for item in items])
        return "No results found."
    except requests.exceptions.RequestException as e:
        return f"Error during web search: {e}"

# Function to log tool usage
def log_tool_usage(tool_name: str, query: str) -> None:
    """
    Log the usage of a tool.
    :param tool_name: Name of the tool being used.
    :param query: The query or input used with the tool.
    """
    if tool_name not in tool_usage_log:
        tool_usage_log[tool_name] = []
    tool_usage_log[tool_name].append(query)
    print(f"Tool '{tool_name}' used with query: {query}")

# Wrapper function with logging
def web_search_with_logging(query: str) -> str:
    """
    Perform a web search and log its usage.
    :param query: The search query string.
    :return: A formatted string of search results or an error message.
    """
    log_tool_usage("Web search tool", query)
    return web_search(query, api_key=google_api_key, cse_id=google_cse_id)
    