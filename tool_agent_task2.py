from langgraph.graph import START, END, StateGraph, MessagesState
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from typing import Literal
import os
from dotenv import load_dotenv
from duckduckgo_search import DDGS


load_dotenv()
# load your api key
open_api_key = os.getenv("OPENAI_API_KEY")

if not open_api_key:
    raise ValueError("No API KEY FOUND! Contact the right authorities")

print("Successfully loaded your API KEY!")

#initialise your model

llm = ChatOpenAI(
    model = "gpt-5-nano",
    temperature= 0.7,
    api_key=open_api_key
)

print(f"Initialized Model: {llm.model_name}")

# creating the three tools: weather, dictionary and web search

@tool
def get_weather(city_name: str) -> str:
    """
    Look at the specified city_name, fetch its weather condition and return it
    Use this tool when you need to report the weather condition of a city_name.

    Args:
         city_name (str): The name of the city (e.g., 'Lagos').

    Returns:
        Weather condition of the city_name(temperature, precipitation)

    Examples:
        - "Lagos" returns 25°C sunny day
        - "Plateau" returns < 10°C cold day
    """
    try:
        if city_name.lower() == "lagos":
            return f"25°C temperature, What a sunny day!"
        elif city_name.lower() == "abeokuta":
            return f"29°C temperature, What a sunny day!"
        elif city_name.lower() == "plateau":
            return f"10°C temperature, It is a very cold day!"
    except Exception as e:
        return f"Could not get the weather condition: {str(e)}"
    
print("Weather tool is ready!")


@tool
def dictionary(word: str) -> str:
    """
    Look at the word and fetch its dictionary meaning 
    Use this tool when you need to check the dictionary.

    Args:
        word (str): The word to check its meaning (e.g., 'Ephemeral').

    Returns:
        The dictionary meaning as a string

    Examples:
        - "Appetite" returns a natural desire to satisfy a bodily need, especially for food
       
    """
    try:
        if word.lower() == "ephemeral":
            return f"(adj) lasting for a very short time. e.g fashions are ephemeral: new ones regularly drive out the old. similar: transient, fleeting"
        elif word.lower() == "euphoria":
            return f"(noun) a feeling or state of intense excitement and happiness. e.g in his euphoria, he had become convinced he could defeat them. similar: joy, elation"
        elif word.lower() == "contentious":
            return f"causing or likely to cause an argument. e.g This is  acontentious issue. similar: controversial, disputable"
    except Exception as e:
        return f"Could not get the dictionary meaning: {str(e)}"
    
print("Dictionary tool is ready!")

@tool
def web_search(query):
    """
    Look at the query and do a web search, return the result found
    Use this tool when you need to do a web search on query.

    Args:
        query: The query you search on the web

    Returns:
        Web search result on the query
    """
    with DDGS() as ddgs:
        results = [r for r in ddgs.text(query, max_results=2)]
        return results


# lets bind our tools to the llm
tools = [get_weather, dictionary, web_search]

tooled_llm = llm.bind_tools(tools)
print(f"LLM bound to {len(tools)} tools")
print(f"  Tools: {[tool.name for tool in tools]}")

# assistant node - that decides when to use a tool or not
sys_msg = SystemMessage(content="""You are a helpful assistant with access to tools.
                    
When asked about the weather of a city, use the get_weather tool.
When asked about the dictionary meaning of a word, use the dictionary tool.
When asked about queries that requires searching the web, use the web_search tool.
                        
Only use tools when necesary - for simple questions, you answer directly                  
""")
def assistant(state: MessagesState) -> dict:
    """
    Assistant node - decides whether to use tools or answer directly
    """

    messages = [sys_msg] + state["messages"]
    response = tooled_llm.invoke(messages)
    return {"messages": [response]}

print("We have our assistant node ready!")

# lets route conditionally
def should_continue(state: MessagesState) -> Literal["tools", "__end__"]:
    """"
    Decides the next step based on the last message

    If LLM called a tool -> go to "tools" node
    If LLM provided final answer -> go to END
    """

    last_msg = state["messages"][-1]

    if last_msg.tool_calls:
        return "tools"
    
    return "__end__"

print("Routed Conditionally!")

# Lets build the graph
builder = StateGraph(MessagesState)
# our node
builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))

# our edges
builder.add_edge(START, "assistant")
builder.add_conditional_edges("assistant", should_continue, {"tools": "tools", "__end__": END})

builder.add_edge("tools", "assistant")

# Adding Memory
memory = MemorySaver()

agent = builder.compile(checkpointer=memory)

print("Our agent is equipped with tools and memory")


# Helper function to run the agent

def run_agent(user_input: str, thread_id: str = "the_agent"):
    """
    Run agent and display the conversation
    """
    print(f"\n{'='*70}")
    print(f"👤 User: {user_input}")
    print(f"{'='*70}\n")

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id": thread_id}}
    )

    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            continue  
        elif isinstance(message, AIMessage):
            if message.tool_calls:
                print(f"🤖 Agent: [Calling tool: {message.tool_calls[0]['name']}]")
            else:
                print(f"🤖 Agent: {message.content}")
        elif isinstance(message, ToolMessage):
            print(f"🔧 Tool Result: {message.content[:100]}..." if len(message.content) > 100 else f"🔧 Tool Result: {message.content}")

    print(f"\n{'='*70}\n")

print("Helper function ready!")