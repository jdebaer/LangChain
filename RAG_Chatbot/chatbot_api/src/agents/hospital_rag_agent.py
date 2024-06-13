import os
from langchain_openai import ChatOpenAI
from langchain.agents import (
    create_openai_functions_agent,
    Tool,
    AgentExecutor,
)
from langchain import hub

# Load the tools that we created.
from chains.hospital_semantic_chain import hospital_semantic_chain
from chains.hospital_cypher_chain import hospital_cypher_chain
from tools.wait_times import (
    get_current_wait_times,
    get_most_available_hospital,
)

# This is the LLM that will act as our agent’s brain, deciding which tools to call and what inputs to pass them.
HOSPITAL_AGENT_MODEL = os.getenv("HOSPITAL_AGENT_MODEL")

# We load a prompt template from LangChain Hub: the default prompt for OpenAI function agents
hospital_agent_prompt = hub.pull("hwchase17/openai-functions-agent")

# Our agent has four tools available to it: Experiences, Graph, Waits, and Availability. The Experiences and Graph tools call .invoke() 
# from their respective chains, while Waits and Availability call the wait time functions you defined. Notice that many of the tool 
# descriptions have few-shot prompts, telling the agent when it should use the tool and providing it with an example of what inputs to pass.

tools = [
    Tool(
        name="Experiences",
        func=hospital_semantic_chain.invoke,
        description="""Useful when you need to answer questions
        about patient experiences, feelings, or any other qualitative
        question that could be answered about a patient using semantic
        search. Not useful for answering objective questions that involve
        counting, percentages, aggregations, or listing facts. Use the
        entire prompt as input to the tool. For instance, if the prompt is
        "Are patients satisfied with their care?", the input should be
        "Are patients satisfied with their care?".
        """,
    ),
    Tool(
        name="Graph",
        func=hospital_cypher_chain.invoke,
        description="""Useful for answering questions about patients,
        physicians, hospitals, insurance payers, patient review
        statistics, and hospital visit details. Use the entire prompt as
        input to the tool. For instance, if the prompt is "How many visits
        have there been?", the input should be "How many visits have
        there been?".
        """,
    ),
    Tool(
        name="Waits",
        func=get_current_wait_times,
        description="""Use when asked about current wait times
        at a specific hospital. This tool can only get the current
        wait time at a hospital and does not have any information about
        aggregate or historical wait times. Do not pass the word "hospital"
        as input, only the hospital name itself. For example, if the prompt
        is "What is the current wait time at Jordan Inc Hospital?", the
        input should be "Jordan Inc".
        """,
    ),
    Tool(
        name="Availability",
        func=get_most_available_hospital,
        description="""
        Use when you need to find out which hospital has the shortest
        wait time. This tool does not have any information about aggregate
        or historical wait times. This tool returns a dictionary with the
        hospital name as the key and the wait time in minutes as the value.
        """,
    ),
]

# Now we instantiate the agent.

chat_model = ChatOpenAI(
    model=HOSPITAL_AGENT_MODEL,
    temperature=0,
)

# This creates an agent that’s been designed by OpenAI to pass inputs to functions. It does this by returning JSON objects that 
# store function inputs and their corresponding value. We need this capability in the LLM for the direct function calls.
hospital_rag_agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=hospital_agent_prompt,
    tools=tools,
)

hospital_rag_agent_executor = AgentExecutor(
    agent=hospital_rag_agent,
    tools=tools,
    return_intermediate_steps=True,
    verbose=False,
)









