import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Start - for RAG
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema.runnable import RunnablePassthrough
# End - for RAG

from framework.chatbot import chat_model

dotenv.load_dotenv()

review_system_prompt_template_str = """Your job is to use patient reviews to answer questions about their experience at a hospital.  Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. If you don't know an answer, say you don't know.

{context}
"""

review_system_prompt_template = SystemMessagePromptTemplate(prompt=PromptTemplate(
										input_variables=['context'],
										template = review_system_prompt_template_str
										))
review_human_prompt_template = HumanMessagePromptTemplate(prompt=PromptTemplate(
										input_variables=['question'],
										template = '{question}'
										))

review_chat_prompt_template = ChatPromptTemplate(
						input_variables = ['context', 'question'],
						messages = [review_system_prompt_template, review_human_prompt_template]
						)

#context = "I had a great stay!"
#question = "Did anyone have a positive experience?"
#print(review_chat_prompt_template.format(context=context, question=question))


# Chain below is final chain without RAG - compare with what follows.
#review_chain = review_chat_prompt_template | chat_model | StrOutputParser()

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











