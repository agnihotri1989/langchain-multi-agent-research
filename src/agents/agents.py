from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from src.tools.tools import web_search, scrape_url

# 1. .env se GROQ_API_KEY load karna
load_dotenv()

# 2. Brain (Groq LLM) initialize karna
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0,
)


# --- AGENT 1: Search Agent (Tavily Web Search Tool ke saath) ---
def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a skilled web research agent. Use the web_search tool to find "
            "the most relevant and recent sources for the given topic. "
            "Return the titles, URLs, and key snippets clearly."
        ),
    )


# --- AGENT 2: Reader Agent (Web Scraper Tool ke saath) ---
def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a deep-reading research agent. From the provided search results, "
            "pick the 2 most relevant URLs and use the scrape_url tool to extract "
            "detailed content from them. Summarize the key extracted facts cleanly."
        ),
    )


# --- AGENT 3: Writer Agent (Modern LCEL Chain) ---
writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert research writer. Write a clean, well-structured, "
        "factual, and insightful research report.",
    ),
    (
        "human",
        """Write a detailed research report on the topic below using the provided research material.

Topic: {topic}

Research Material:
{research}

Structure the report clearly with:
1. Introduction
2. Key Findings (minimum 3 well-explained points)
3. Conclusion
4. Sources (list all relevant URLs found in the research material)

Be detailed, factual, and professional.""",
    ),
])

writer_chain = writer_prompt | llm | StrOutputParser()


# --- AGENT 4: Critic Agent (Modern LCEL Chain) ---
critic_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a sharp, constructive senior research critic. Be honest and specific.",
    ),
    (
        "human",
        """Review the research report below and evaluate it strictly.

Report:
{report}

Provide your evaluation in this exact structure:
- Score: X/10
- Strengths: (bullet points)
- Areas to Improve: (bullet points)
- One-Line Verdict:""",
    ),
])

critic_chain = critic_prompt | llm | StrOutputParser()
