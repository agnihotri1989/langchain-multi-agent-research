import os
import re
import requests
import trafilatura
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient

# .env file se TAVILY_API_KEY load karna
load_dotenv()
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def web_search(query: str) -> str:
    """Search the live internet for recent, relevant information and return top 5 titles, URLs, and snippets."""
    result = tavily.search(query=query, max_results=5)

    formatted_results = []
    for item in result.get("results", []):
        title = item.get("title", "No Title")
        url = item.get("url", "")
        snippet = item.get("content", "")[:300]  # Sirf 300 chars snippet taaki tokens bachein
        formatted_results.append(
            f"Title: {title}\nURL: {url}\nSnippet: {snippet}"
        )

    return "\n\n---\n\n".join(formatted_results)


@tool
def scrape_url(url: str) -> str:
    """Fetch and extract clean main text content from a given webpage URL."""
    try:
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # Method 1: Trafilatura (best for clean article extraction)
        extracted = trafilatura.extract(response.text)

        # Method 2: Fallback to BeautifulSoup agar trafilatura kuch na nikaal paye
        if not extracted:
            soup = BeautifulSoup(response.text, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            extracted = soup.get_text(separator=" ", strip=True)

        # Extra spaces clean karna aur max 4000 characters tak limit rakhna (LLM token safety!)
        clean_text = re.sub(r"\s+", " ", extracted).strip()
        return clean_text[:4000] if clean_text else "No readable content found."

    except Exception as e:
        return f"Failed to scrape {url}: {str(e)}"
