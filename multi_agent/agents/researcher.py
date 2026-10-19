import operator
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import Annotated, List
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_tavily import TavilySearch, TavilyExtract
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import add_messages, StateGraph, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool, InjectedToolCallId
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command
from datetime import datetime
import os

from google import genai


load_dotenv()


researcher_prompt= open(r"langgraph_agent\multi_agent\prompts\researcher.md","r").read()

api_key= os.getenv("google_gemini_api_key")
# client = genai.Client(api_key=api_key)
model = ChatGoogleGenerativeAI(model= "gemini-3.5-flash-lite", api_key= api_key)
# messages= [
#     "system","You are an helpful ai assistant",
#     "user", "Hi"
# ]
# response= model.invoke(messages)

# print(response)

@tool
async def search_web(
    query: str,
    num_results: int =3
) -> dict :
    """
    Search the web and get back a list of search result including the page title, url and short summary of each webpage.

    Args:
        query: query to search on the web.
        num_results: The number of results to return, max is 3.

    return:
        a dictionary of search results
    """
    web_search = TavilySearch(max_results= min(num_results, 3), topic= "general")
    search_results= web_search.invoke(input= {"query":query})

    processed_results = {
        "query":query,
        "results":[]
    }

    for result in search_results['results']:
        processed_results['results'].append({
            "title": result["title"],
            "url": result["url"],
            "content_preview": result["content"]
        })

    return processed_results

@tool
async def extract_content_from_webpage(urls:List[str]):
    """
    Extract content from a webpage.
    Args:
        lisr of urls of the webpages to extract the content from
    
    return:
        A list of dictionaries containing the extracted content from each webpage
    """
    web_extract = TavilyExtract()
    result= web_extract.invoke(input= {"urls":urls})["result"]
    return result

class ResearchReport(BaseModel):
    topic: str
    report: str

@tool 
async def generate_research_report(
    topic: str, 
    report: str,
    tool_call_id: Annotated[str, InjectedToolCallId]
    ):
        """
        Generate a research report on specific topic.

        Args:
            topic: The topic to research
            report: The research report
        
        Returns:
            The research report.
        """
        #model_validate() is a class method that takes a standard Python object (like a dictionary) and validates it against your model schema to return a clean, typed model instance
        research_report = ResearchReport.model_validate({
            "topic": topic,
            "report": report
        })


        #We use command premitive to update the state with research report and add a tool message to the conversation with generated report
        # Command allows node or tool to update state nad direct execution to another node
        #model_dump_json() is an instance method that serializes your existing model instance directly into a JSON-formatted string.
        return Command(update={
            "research_reports": [research_report],
            "messages":[ToolMessage(
                name= "generate_research_report",
                content= research_report.model_dump_json(),
                tool_call_id = tool_call_id,
            )],
        })
        #it allows the destination to be determined at runtime, based on the chosen tool arguments, rather than requiring every possible destination to be encoded as a fixed edge from the tool node.

class ResearcherState(BaseModel):
    """
    The state of the researcher agent.

    The research_reports attribute is shared with the supervisor state. This allows the supervisor to access the research_reports of the research agent and share them with copywriter agent.
    """
    #Annotated adds extra metadata such that the langgraph should use add_messages when applying updates to this field
    messages: Annotated[list, add_messages]= []

    #operator.add tells LangGraph to combine the existing list with the list returned by a node.
    #This is useful if the workflow performs several research tasks and needs to accumulate their results.
    research_reports: Annotated[list, operator.add]= []

tools=[
    search_web,
    extract_content_from_webpage,
    generate_research_report,
]


llm_with_tools= model.bind_tools(tools)

async def researcher(state: ResearcherState):
    "The main researcher agent."
    response = llm_with_tools.invoke([
                SystemMessage(content= researcher_prompt.format(current_datetime= datetime.now()))
    ] + state.messages)
    return {"messages": [response]}

builder= StateGraph(ResearcherState)

builder.add_node("researcher",researcher)
builder.add_node("researcher_toolnode", ToolNode(tools))

builder.set_entry_point("researcher")
builder.add_conditional_edges(
    "researcher",
    tools_condition,{
        "tools":"researcher_toolnode",
        END:END
    }
)
builder.add_edge("researcher_toolnode", "researcher")

graph= builder.compile()

print(graph.get_graph().edges)








