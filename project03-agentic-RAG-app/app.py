import streamlit as st
from app.llm import load_llm
from app.vectorstore import build_vectorstore
from app.tools import create_tools
from app.agents import create_agents
from crewai import Task

@st.cache_resource
def load_vectorstore():
    return build_vectorstore()

st.title("🧠 Agentic RAG App")

query = st.text_input("Enter your question:")

if st.button("Run Agent"):
    with st.spinner("Running agents..."):
        llm = load_llm()
        vectorstore = load_vectorstore()
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # tools = create_tools(retriever)
        researcher, writer, critic = create_agents(llm)

        # === Step 1: Call RAG manually ===
        local_docs = retriever.get_relevant_documents(query)
        local_context = "\n\n".join([doc.page_content for doc in local_docs[:2]])

        # === Step 2: Call Tavily manually ===
        from tavily import TavilyClient
        from app.config import TAVILY_API_KEY

        tavily_client = TavilyClient(api_key=TAVILY_API_KEY)
        web_results = tavily_client.search(query=query, max_results=1)

        web_context = web_results["results"][0]["content"]

        combined_context = f"""
                Local Knowledge:
                {local_context}

                Web Knowledge:
                {web_context}
        """
        combined_context = combined_context[:3000]

        # === Step 3: Researcher summarizes context ===
        research_task = Task(
            description=f"""
            Answer the user's question strictly using the context below.

            Question:
            {query}

            Context:
            {combined_context}

            Provide a concise and accurate answer.
            """,
            expected_output="Direct answer to the user's question",
            agent=researcher
        )
        research_summary = researcher.execute_task(research_task)

        # === Step 4: Writer creates report ===
        write_task = Task(
            description=f"""
            Write a structured professional report that answers the user's question.

            User Question:
            {query}

            Research Summary:
            {research_summary}

            The report must clearly address the question.
            """,
            expected_output="Structured report answering the question",
            agent=writer
        )
        draft_report = writer.execute_task(write_task)

        # === Step 5: Critic refines ===
        review_task = Task(
            description=f"""
            Improve the clarity, grammar, and coherence of the following report.
            Do NOT change the topic.
            Do NOT introduce new content.
            Keep the original meaning and focus strictly on the user question.

            Report:
            {draft_report}
            """,
            expected_output="Refined report with same meaning",
            agent=critic
        )
        final_report = critic.execute_task(review_task)

        st.session_state.result = final_report

if "result" in st.session_state:
    st.write(st.session_state.result)