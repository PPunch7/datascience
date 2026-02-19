from crewai import Task, Crew

def create_crew(researcher, writer, query):

    research_task = Task(
        description=f"Research the following topic: {query}",
        expected_output="Research findings.",
        agent=researcher
    )

    write_task = Task(
        description="Write a detailed report based on the research findings.",
        expected_output="Structured report.",
        agent=writer
    )

    return Crew(
        agents=[researcher, writer],
        tasks=[research_task, write_task],
        verbose=False
    )