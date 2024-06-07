from fastapi import FastAPI
from agents.hospital_rag_agent import hospital_rag_agent_executor
from models.hospital_rag_query import HospitalQueryInput, HospitalQueryOutput
from utils.async_utils import async_retry

app = FastAPI(
    title="Hospital Chatbot",
    description="Endpoints for a hospital system graph RAG chatbot",
)

# The @async_retry decorator ensures that invoke_agent_with_retry() will be retried ten times with a delay of one second before failing.
@async_retry(max_retries=10, delay=1)
async def invoke_agent_with_retry(query: str):
    """Retry the agent if a tool fails to run.

    This can help when there are intermittent connection issues
    to external APIs.
    """
    return await hospital_rag_agent_executor.ainvoke({"input": query})

@app.get("/")
async def get_status():
    return {"status": "running"}

# query_hospital_agent() serves POST requests to our agent at /hospital-rag-agent. This function extracts the text field from the request body, 
# passes it to the agent, and returns the agent’s response to the user.
@app.post("/hospital-rag-agent")
async def query_hospital_agent(query: HospitalQueryInput) -> HospitalQueryOutput:
    query_response = await invoke_agent_with_retry(query.text)
    query_response["intermediate_steps"] = [
        str(s) for s in query_response["intermediate_steps"]
    ]

    return query_response
