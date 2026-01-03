from langgraph.graph import START, END, StateGraph, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
import os


load_dotenv()
# load api key
open_api_key = os.getenv("OPENAI_API_KEY")

if not open_api_key:
    raise ValueError("API KEY not found!. Do the needful")

print("Successfully loaded your API KEY")

# initialise your model

llm = ChatOpenAI(
    model="gpt-5-nano",
    temperature=0.7,
    api_key=open_api_key
)
print(f"Model ready: {llm.model_name}")

# creating my node
sys_msg = SystemMessage(
    content= "You are a friendly and helpful customer support representative. Be concise"
)

def assistant(state: MessagesState) -> dict:
    """
    The assistant node - processes messages and generates response
    """
    # combine the prompt with the convo history
    messages = [sys_msg] + state["messages"]

    #Get model response
    response = llm.invoke(messages)

    # which is then returned as an updated state
    return {
        "messages": [AIMessage(content=response.content)]
    }
    
print("Assistant node defined")

# our StateGraph with MessagesState
builder = StateGraph(MessagesState)

#add the assistant node
builder.add_node("assistant", assistant)

# Defining the flow
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

print("We have our Graph structured")

# This is creating the checkpointer that stores in memory
memory = MemorySaver()

# compile the graph with memory
agent = builder.compile(checkpointer=memory)

print("Now our agent has been compiled with memory")

# Okay let us run the agent 

session_id = "Sales-001"
print(f"Our convo is from this session_id: {session_id}")

def run_convo(user_input: str, thread_id: str = session_id):
    """
    Docstring for run_convo
    
    :param user_input: Description
    :type user_input: str
    :param thread_id: Description
    :type thread_id: str

    Sends message to agent and get response, thread_id should be different for each users
    """

    #invoke the spirit of the agent

    result = agent.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config={"configurable": {"thread_id":thread_id}}
    )

    for message in result["messages"]:
        if isinstance(message, HumanMessage):
            print(f"\n Customer: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\n Agent: {message.content}")

    print("\n" + "="*70)

print("I am ready, Let's Go!")