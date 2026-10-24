import asyncio
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import InMemorySaver

from dotenv import load_dotenv
from pydantic import BaseModel, Field
from typing import Optional, Annotated
from typing_extensions import TypedDict

from cadence.discovery.models import BookCandidate
from  cadence.discovery.catalog import get_book_details

load_dotenv()

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    raw_query: str
    cleaned_title: str
    author: Optional[str]
    book_results: list


class BookSearchQuery(BaseModel):
    cleaned_title: str = Field(
        description="The clean, isolated title of the book extracted from the user query. Remove conversational fluff or descriptions."
    )
    author: Optional[str] = Field(
        default=None,
        description="The name of the author of the book if mentioned in query. Otherwise null."
    )

llm = ChatGoogleGenerativeAI(
    model="gemini-3.0-flash"
)
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are an expert library search assistant. Analyze the user's raw search phrase, extract the actual book title and author name, and isolate them from conversational fluff or background descriptions."),
    ("user", "Extract the book title and author from this query: '{raw_query}'")
])

structured_llm = prompt_template | llm.with_structured_output(BookSearchQuery)

def display_book_info(book_candidate: BookCandidate):
    print(f"\nTitle:   {book_candidate.title}")
    print(f"Author:  {book_candidate.authors}")
    print(f"Link:    {book_candidate.meta.link}")
    print("Metadata:")
    print(f"   • First Published:  {book_candidate.meta.first_publish_year}")
    print(
        f"   • Total Editions:   {book_candidate.meta.total_editions_cataloged}"
    )
    print(f"   • Key Subjects:     {', '.join(book_candidate.meta.top_subjects)}")


@tool
async def discovery_tool(cleaned_title: str, author: Optional[str] = None) -> str:
    """
    Search the library catalog for a book using its cleaned title and optional author name.
    Use this tool when the user is trying to find or lookup book details.
    
    Args:
        cleaned_title: The isolated name of the book.
        author: The name of the author (optional).
    """

    try:
        search_description = f"Searching for books with title '{cleaned_title}'"
        if author:
            search_message += f" by '{author}'"

        result = await get_book_details(cleaned_title)

        if not result:
            return "\nNo books found matching that query."

        if isinstance(result, list) and len(result) > 0:
            output = f"\nFound {len(result)} book matches."
            for index, book in enumerate(result, start=1):
                output += f"[{index}] Title: {book.title} -by {', '.join(book.authors)}\n"
            return output

        elif isinstance(result, BookCandidate):
            return f"Found Book Match:\nTitle: {result.title}\nAuthor: {', '.join(result.authors)}\nLink: {result.meta.link}"
        
        return str(result)

    except Exception as e:
        print(f"An error occurred: {e}")


tools = [discovery_tool]

tool_node = ToolNode(tools)


def chat_node(state: State):
    llm_with_tools = llm.bind_tools([discovery_tool])
    response = llm_with_tools.invoke(state['messages'])
    return {"messages": [response]}


def query_node(state: State):
    last_query = state['messages'][-1].content
    extraction = structured_llm.invoke({
        'raw_query': last_query,
        'messages': state['messages']
        })

    return{
        "raw_query": last_query,
        "cleaned_title": extraction.cleaned_title,
        "author": extraction.author
    }


async def discovery_node(state: State) -> dict:
    title = state.get('cleaned_title')
    author = state.get('author')

    print(f"\n[System] Searching for books with title '{title}'" + (f" by '{author}'" if author else ""))

    result = await get_book_details(title)

    book_list = []
    if isinstance(result, list):
        book_list = result
    elif result:
         book_list = [result]

    return {'book_results': book_list}


async def selection_node(state: State) -> dict:
    books = state.get("book_results", [])

    if not books:
        print("\nNo books found matching that query.")
        return {}

    if len(books) == 1:
        display_book_info(books[0])
        return {}

    print(f"\nFound {len(books)} book matches.")
    for index, book in enumerate(books, start=1):
        print(f"[{index}] {book.title} -by {', '.join(book.authors)}")

    while True:
        selection = await asyncio.to_thread(input, "\nEnter the number of the book to view details (or 'c' to cancel): ")
        selection = selection.strip().lower()
        
        if selection == 'c':
            print("Selection cancelled.")
            break
            
        if selection.isdigit():
            choice_idx = int(selection) - 1
            if 0 <= choice_idx < len(books):
                display_book_info(books[choice_idx])
                break
                
        print("Invalid selection. Please enter a valid number or 'c' to cancel.")
        
    return {}

def download_approval_node(state: State):
    pass


def download_result_node(state: State):
    pass

checkpointer = InMemorySaver()


graph = StateGraph(State)
graph.add_node('chat_node', chat_node)
graph.add_node('tools', tool_node)
graph.add_node('query_node', query_node)
graph.add_node('discovery_node', discovery_node)
graph.add_node('selection_node', selection_node)
graph.add_node('download_approval_node', download_approval_node)
graph.add_node('download_result_node', download_result_node)

graph.add_edge(START, 'chat_node')
graph.add_edge('chat_node', 'query_node')
graph.add_edge('query_node', 'discovery_node')
graph.add_edge('discovery_node', 'selection_node')
graph.add_edge('selection_node', END)

app = graph.compile(checkpointer=checkpointer)


