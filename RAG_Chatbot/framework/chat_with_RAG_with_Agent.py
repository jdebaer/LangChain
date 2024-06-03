import dotenv
from langchain_core.output_parsers import StrOutputParser

# Start - for RAG
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
# End - for RAG

# Start - for Agent
from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain import hub
# End - for Agent

from framework.chatbot import chat_model
from framework.review_chat_prompt_template import review_chat_prompt_template
from framework.tools import get_current_wait_time, get_tools


dotenv.load_dotenv()
#dotenv.config({path: "./vars/.env"}

REVIEWS_CHROMA_PATH = "chroma_data/"

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings()
)

reviews_retriever  = reviews_vector_db.as_retriever(k=10)

review_chain = (
    {"context": reviews_retriever, "question": RunnablePassthrough()}
    | review_chat_prompt_template
    | chat_model
    | StrOutputParser()
)

# New stuff for Agent-based approach goes here.

tools = get_tools(review_chain)

# We don't use the template we created ourselves, but an openAI one. There is an input parameter called 'input'.
agent_chat_prompt_template = hub.pull("hwchase17/openai-functions-agent")

agent = create_openai_functions_agent(
    llm=chat_model,
    prompt=agent_chat_prompt_template,
    tools=tools,
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    return_intermediate_steps=False,
    verbose=False,
)









