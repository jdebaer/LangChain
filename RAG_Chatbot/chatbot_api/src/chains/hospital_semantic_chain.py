import os
from langchain.vectorstores.neo4j_vector import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate

HOSPITAL_QA_MODEL = os.getenv("HOSPITAL_QA_MODEL")

# The code below adds semantic embeddings to the reviews we stored in the graph database. The embedding is called 'embedding'.
# Once Neo4jVector.from_existing_graph() runs, you’ll see that every Review node in Neo4j has an embedding property which is 
# a vector representation of the physician_name, patient_name, text, and hospital_name properties. This allows you to answer 
# questions like Which hospitals have had positive reviews? It also allows the LLM to tell you which patient and physician wrote 
# reviews matching your question.

neo4j_vector_index = Neo4jVector.from_existing_graph(
    # The model used to create the embeddings—we’re using OpenAIEmeddings() in this example.
    embedding=OpenAIEmbeddings(),
    url=os.getenv('NEO4J_URI'),
    username=os.getenv('NEO4J_USERNAME'),
    password=os.getenv('NEO4J_PASSWORD'),
    # The name given to the vector index.
    index_name='reviews',
    # The node to create the embeddings for.
    node_label='Review',
    # The node properties to include in the embedding. Any piece of information that you want the LLM to be able to use when answering the queary
    # must be part of the embedding, so that it will be fed into the prompt.
    text_node_properties=[
        'physician_name',
        'patient_name',
        'text',
        'hospital_name',
    ],
    # The name of the embedding node property.
    embedding_node_property='embedding',
)

# Craft the chain, starting with the prompt template.

system_prompt_template_text = """Your job is to use patient reviews to answer questions about their experience at a hospital. Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. If you don't know an answer, say you don't know.
{context}
"""

system_prompt_template = SystemMessagePromptTemplate(prompt=PromptTemplate(input_variables=["context"], template=system_prompt_template_text))

human_prompt_template = HumanMessagePromptTemplate(prompt=PromptTemplate(input_variables=["question"], template="{question}"))

chat_prompt_template = ChatPromptTemplate(input_variables=["context", "question"], messages=[system_prompt_template, human_prompt_template])

hospital_semantic_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model=HOSPITAL_QA_MODEL, temperature=0),
    # Chain type 'stuff' means telling the chain to pass all 12 reviews to the prompt.
    chain_type="stuff",
    retriever=neo4j_vector_index.as_retriever(k=12),
)

hospital_semantic_chain.combine_documents_chain.llm_chain.prompt = chat_prompt_template





