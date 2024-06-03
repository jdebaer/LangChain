
# Run this from the project's root directory, so one above framework.

import dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

REVIEWS_CHROMA_PATH = "chroma_data/"

dotenv.load_dotenv()

reviews_vector_db = Chroma(
    persist_directory=REVIEWS_CHROMA_PATH,
    embedding_function=OpenAIEmbeddings(),
)

# Here we do a search on semantics - we find reviews that match this specific questions without grepping for words etc.

question = """Has anyone complained about communication with the hospital staff?"""
relevant_docs = reviews_vector_db.similarity_search(question, k=3)
  
print(len(relevant_docs))
print(relevant_docs[0].page_content)
