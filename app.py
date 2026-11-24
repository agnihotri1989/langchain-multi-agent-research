import streamlit as st
from src.agents.agents import (
    build_search_agent,
    build_reader_agent,
    writer_chain,
    critic_chain,
)

# 1. Page Configuration
st.set_page_config(
    page_title="Researcher Agent | Multi-Agent AI System",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Researcher Agent — Multi-Agent AI System")
st.markdown(
    "4 specialized AI agents collaborate — **Searching, Scraping, Writing, and Critiquing** — "
    "to deliver a polished research report on any topic."
)

# 2. Quick Example Topics (Video [01:24:44])
examples = [
    "The impact of AI on the job market in 2026",
    "AGI development in the next 5 years",
    "Top latest AI Agent frameworks in 2026",
]

st.markdown("**Try an example:**")
cols = st.columns(len(examples))
for idx, ex in enumerate(examples):
    if cols[idx].button(ex, use_container_width=True):
        st.session_state["topic_input"] = ex

# 3. Topic Input Box
topic = st.text_input(
    "Enter your research topic:",
    value=st.session_state.get("topic_input", ""),
    placeholder="e.g., The impact of AI on the job market in 2026",
)

# 4. Run Research Pipeline Button
if st.button("🚀 Run Research Pipeline", type="primary"):
    if not topic.strip():
        st.warning("Please enter a research topic first!")
    else:
        state = {"topic": topic}

        # Live Status Box (Video [01:25:12])
        with st.status("Running Multi-Agent Research Pipeline...", expanded=True) as status:
            # Step 1: Search Agent
            st.write("🔍 **Agent 1 (Search Agent):** Searching live internet via Tavily...")
            search_agent = build_search_agent()
            search_res = search_agent.invoke({
                "messages": [{"role": "user", "content": f"Find recent, reliable, and detailed information about: {topic}"}]
            })
            state["search_result"] = search_res["messages"][-1].content
            st.write("✅ Search completed!")

            # Step 2: Reader Agent
            st.write("📖 **Agent 2 (Reader Agent):** Scraping top URLs for deep content...")
            reader_agent = build_reader_agent()
            reader_res = reader_agent.invoke({
                "messages": [{
                    "role": "user",
                    "content": (
                        f"Based on the following search results for '{topic}', "
                        f"pick the most relevant URLs and scrape them for deeper content:\n\n"
                        f"{state['search_result']}"
                    ),
                }]
            })
            state["scraped_content"] = reader_res["messages"][-1].content
            st.write("✅ Scraping completed!")

            # Step 3: Writer Agent
            st.write("✍️ **Agent 3 (Writer Agent):** Drafting structured research report...")
            combined_research = (
                f"### Search Results & Sources:\n{state['search_result']}\n\n"
                f"### Detailed Scraped Content:\n{state['scraped_content']}"
            )
            state["report"] = writer_chain.invoke({
                "topic": topic,
                "research": combined_research,
            })
            st.write("✅ Report drafted!")

            # Step 4: Critic Agent
            st.write("🧐 **Agent 4 (Critic Agent):** Reviewing and scoring the report...")
            state["feedback"] = critic_chain.invoke({"report": state["report"]})
            st.write("✅ Review completed!")

            status.update(label="🎉 All 4 Agents Completed Successfully!", state="complete")

        # 5. Display Results (Expandable Agent Outputs + Final Report + Download + Critic)
        st.divider()

        with st.expander("🔍 Agent 1 Output: Search Results & URLs"):
            st.markdown(state["search_result"])

        with st.expander("📖 Agent 2 Output: Scraped Deep Content"):
            st.markdown(state["scraped_content"])

        st.subheader("📄 Final Research Report")
        st.markdown(state["report"])

        # Download as Markdown button (Video [01:27:33])
        st.download_button(
            label="⬇️ Download Report (.md)",
            data=state["report"],
            file_name=f"{topic.lower().replace(' ', '_')[:40]}_report.md",
            mime="text/markdown",
        )

        st.divider()
        st.subheader("🧐 Critic Agent Feedback")
        st.info(state["feedback"])
