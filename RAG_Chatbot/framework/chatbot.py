import dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

dotenv.load_dotenv()

chat_model = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0)


## Quick test without templating - comment out.
#from langchain.schema.messages import HumanMessage, SystemMessage
#
#messages = [
#    SystemMessage(
#        content="""You're an assistant knowledgeable about
#        healthcare. Only answer healthcare-related questions."""
#    ),
#    HumanMessage(content="What is the marketplace?"),
#]
#print(chat_model.invoke(messages))

## Quick test with basic template - comment out.
#from langchain.prompts import ChatPromptTemplate
#
#review_chat_prompt_template_str = """Your job is to use patient reviews to answer questions about their experience at a hospital.  Use the following context to answer questions. Be as detailed as possible, but don't make up any information that's not from the context. If you don't know an answer, say you don't know.
#
#{context}
#
#{question}
#"""
#
## from_messages() is alternative. Note that our string defaults to Human message.
#review_chat_prompt_template = ChatPromptTemplate.from_template(review_chat_prompt_template_str)
#
#context = "I had a great stay!"
#question = "Did anyone have a positive experience?"
#
#print(review_chat_prompt_template.format(context=context, question=question))





























