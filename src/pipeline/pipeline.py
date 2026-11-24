from rich import print
from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)


def run_research_pipeline(topic: str) -> dict:
    # Shared State Memory (temporary dictionary)
    state = {"topic": topic}

    # --- STEP 1: Search Agent ---
    print("\n[bold cyan]" + "=" * 50 + "[/bold cyan]")
    print("[bold cyan]🔍 STEP 1: Search Agent is working...[/bold cyan]")
    print("[bold cyan]" + "=" * 50 + "[/bold cyan]")

    search_agent = build_search_agent()
    search_response = search_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": f"Find recent, reliable, and detailed information about: {topic}",
            }
        ]
    })
    state["search_result"] = search_response["messages"][-1].content
    print(state["search_result"])

    # --- STEP 2: Reader Agent ---
    print("\n[bold yellow]" + "=" * 50 + "[/bold yellow]")
    print("[bold yellow]📖 STEP 2: Reader Agent is scraping top URLs...[/bold yellow]")
    print("[bold yellow]" + "=" * 50 + "[/bold yellow]")

    reader_agent = build_reader_agent()
    reader_response = reader_agent.invoke({
        "messages": [
            {
                "role": "user",
                "content": (
                    f"Based on the following search results for '{topic}', "
                    f"pick the most relevant URLs and scrape them for deeper content:\n\n"
                    f"{state['search_result']}"
                ),
            }
        ]
    })
    state["scraped_content"] = reader_response["messages"][-1].content
    print(state["scraped_content"])

    # --- STEP 3: Writer Chain (LCEL) ---
    print("\n[bold green]" + "=" * 50 + "[/bold green]")
    print("[bold green]✍️ STEP 3: Writer Agent is drafting the report...[/bold green]")
    print("[bold green]" + "=" * 50 + "[/bold green]")

    combined_research = (
        f"### Search Results & Sources:\n{state['search_result']}\n\n"
        f"### Detailed Scraped Content:\n{state['scraped_content']}"
    )
    state["report"] = writer_chain.invoke({
        "topic": topic,
        "research": combined_research,
    })
    print(state["report"])

    # --- STEP 4: Critic Chain (LCEL) ---
    print("\n[bold magenta]" + "=" * 50 + "[/bold magenta]")
    print("[bold magenta]🧐 STEP 4: Critic Agent is reviewing the report...[/bold magenta]")
    print("[bold magenta]" + "=" * 50 + "[/bold magenta]")

    state["feedback"] = critic_chain.invoke({
        "report": state["report"],
    })
    print(state["feedback"])

    return state
