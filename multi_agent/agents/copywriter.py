from datetime import datetime
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, add_messages
from langgraph.prebuilt import ToolNode, InjectedState, tools_condition
from typing import Annotated
from pydantic import BaseModel
from dotenv import load_dotenv
import operator
import os

load_dotenv()

copywriter_prompt = open("langgraph_agent/multi_agent/prompts/copywriter.md", "r").read()
linkedin_example= open("langgraph_agent/multi_agent/example_content/linkedin.md", "r").read()
blog_example= open("langgraph_agent/multi_agent/example_content/blog.md", "r").read()

api_key= os.getenv("google_gemini_api_key")

llm = ChatGoogleGenerativeAI(model= "gemini-3.5-flash-lite", api_key= api_key)

class CopyWriterState(BaseModel):
    """
    The state of copywriter agent.
    The research_reports attribute is shared with the supervisor state. This allows the supervisor to access the research reports geenrated by the researcher and share them with copywriter.
    """
    messages: Annotated[list, add_messages]= []
    research_reports: Annotated[list, operator.add]= []


# IngectedState- The tool need access to the copywriter's current graph state. Rather than askig the llm to provide the research reports
# as arguments, the tool receives the state from langgraph
@tool
async def review_research_reports(
    state: Annotated[CopyWriterState, InjectedState]
):
    """
    Use this tool to review available research reports to inform your writing.

    Returns:
        A list of research reports.
    """
    return [report.model_dump_json() for report in state.research_reports]

@tool
async def generate_linkedin_post(
    title: str,
    content: str
):
    """
    Use this tool to generate the linkedin post

    Args:
        title: The title of the post.
        content: The content of the post in markdown format.

    Returns:
        A string indicating the location of saved post.
    """
    filename = f"ai_files/{title}.md"
    with open(filename, "w") as f:
        f.write(content)
    
    return f"The linkedin post has been generated and saved to {filename}"

@tool
async def generate_blog_post(
    title: str,
    content: str,
):
    """Use this tool to generate a blog post.
    
    Args:
        title: The title of the post.
        content: The content of the post in markdown format.

    Returns:
        A string indicating the location of the saved post.
    """
    filename=f"ai_files/{title}.md"
    with open(filename, "w") as f:
        f.write(content)

    return f"The blog post has been generated and saved to {filename}"

tools= [
    review_research_reports,
    generate_linkedin_post,
    generate_blog_post
]

llm_with_tools = llm.bind_tools(tools)

async def copywriter(state: CopyWriterState):
    """
    The main copywriter agent
    """
    system_prompt= SystemMessage(content= copywriter_prompt.format(
        current_datetime= datetime.now(),
        linkedin_example=  linkedin_example,
        blog_example= blog_example
    ))
    response = llm_with_tools.ainvoke([system_prompt]+ state.messages)
    return {"messages":[response]}

builder = StateGraph(CopyWriterState)

builder.add_node(copywriter)
builder.add_node("copywriter_toolnode", ToolNode(tools))

builder.set_entry_point("copywriter")
builder.add_conditional_edges(
    "copywriter", tools_condition,{
        "tools": "copywriter_toolnode",
        END:END
    }
)
builder.add_edge("copywriter_toolnode", "copywriter")

graph= builder.compile()

print(graph.get_graph().edges)


