from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

messages = [
    SystemMessage("You are an expert in social media content strategy"),
    HumanMessage("Give a suggestion on which social media app to start with as a beginner")
]

result = llm.invoke(messages)

print(result.content)


