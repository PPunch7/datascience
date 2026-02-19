from crewai import Agent

def create_agents(llm):
    # local_tool, web_tool = tools

    researcher = Agent(
        role="Healthcare AI Researcher",
        goal="Provide accurate answers strictly based on given context.",
        backstory="An expert researcher in AI for healthcare.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=1
    )

    writer = Agent(
        role="Professional Medical Writer",
        goal="Write clear and structured healthcare reports.",
        backstory="An expert in transforming technical summaries into readable reports.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=1
    )

    critic = Agent(
        role="Medical Report Reviewer",
        goal="Improve clarity, coherence, and accuracy.",
        backstory="A strict healthcare quality reviewer.",
        llm=llm,
        verbose=False,
        allow_delegation=False,
        max_iter=1
    )

    return researcher, writer, critic