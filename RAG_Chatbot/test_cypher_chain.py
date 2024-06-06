import dotenv
dotenv.load_dotenv('./vars/.env')

from chatbot_api.src.chains.hospital_cypher_chain import (
hospital_cypher_chain
)

question = """What is the average visit duration for emergency visits in North Carolina?"""
print(hospital_cypher_chain.invoke(question))

# Cypher chain does:
# 1. Accept a user's natural language query.
# 2. Convert the NL query to a Cypher query using a first model.
# 3. Run the Cypher query against Neo4j
# 4. Use the Cypher query result and the original NL queary in a prompt for a question answering LLM (second model).
# 5. Return the response. 

