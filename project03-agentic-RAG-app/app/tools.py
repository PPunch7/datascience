from crewai_tools import tool
from tavily import TavilyClient
from .config import TAVILY_API_KEY

tavily_client = TavilyClient(api_key=TAVILY_API_KEY)

def create_tools(retriever):

    @tool
    def local_rag_search(query: str) -> str:
        """
        Search the local PDF knowledge base.
        """
        docs = retriever.invoke(query)[:1]
        return docs[0].page_content

    @tool
    def tavily_search_tool(query: str) -> str:
        """
        Search the web and return summarized results.
        """
        results = tavily_client.search(query=query, max_results=1)
        texts = [r["content"][:500] for r in results["results"]]
        return "\n\n".join(texts)

    return local_rag_search, tavily_search_tool