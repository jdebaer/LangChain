#from framework.chatbot import review_chain
#from framework.chat_with_RAG import review_chain
#from framework.chat_with_RAG_without_Agent import review_chain

# When working with an Agent, we import the executor, not the chain.
from framework.chat_with_RAG_with_Agent import agent_executor


#question = "Has anyone complained about the food?"
question = "What is the wait time at hospital A?"

print(agent_executor.invoke({'input': question})['output'])
