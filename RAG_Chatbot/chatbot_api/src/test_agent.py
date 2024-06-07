import dotenv
dotenv.load_dotenv('../../vars/.env')

from agents.hospital_rag_agent import hospital_rag_agent_executor

response = hospital_rag_agent_executor.invoke(
    {"input": "What is the wait time at Wallace-Hamilton?"}
)
print(response.get("output"))

response = hospital_rag_agent_executor.invoke(
    {"input": "Which hospital has the shortest wait time?"}
)
print(response.get("output"))

# Notice how we never explicitly mention reviews or experiences in our question. The agent knows, based on the tool description, that it needs to 
# invoke Experiences.
response = hospital_rag_agent_executor.invoke(
    {
        "input": (
            "What have patients said about their "
            "quality of rest during their stay?"
        )
    }
)
print(response.get("output"))

response = hospital_rag_agent_executor.invoke(
    {
        "input": (
            "Which physician has treated the "
            "most patients covered by Cigna?"
        )
    }
)
print(response.get("output"))

