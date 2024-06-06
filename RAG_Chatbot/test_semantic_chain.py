import dotenv

#dotenv.config({path: './vars/.env'})
dotenv.load_dotenv('./vars/.env')

from chatbot_api.src.chains.hospital_semantic_chain import hospital_semantic_chain

query = """What have patients said about hospital efficiency? Mention details from specific reviews."""

# In this example, notice how specific patient and hospital names are mentioned in the response. This happens because you embedded hospital and patient 
# names along with the review text, so the LLM can use this information to answer questions.

response = hospital_semantic_chain.invoke(query)

# Semantic chain does:
# 1. Transform query to embedding via embedding model (this is not the LLM we use to answer the query).
# 2. Uses embedding to retrieve semantic matches.
# 3. Put these matches in the prompt and invokes the query-answering LLM.

print(response.get("result"))
