from langchain_openai import ChatOpenAI
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
import os

load_dotenv()

messages = [
    SystemMessage(content="Solve the following math problems"),
    HumanMessage(content="What is the square root of 55"),
]

hf = HuggingFaceEndpoint(
    repo_id="mistralai/Mistral-7B-Instruct-v0.3",  # or "meta-llama/Llama-3.1-8B-Instruct"
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    max_new_tokens=128,
    temperature=0.0,
)

llm_hf = ChatHuggingFace(llm=hf)
#llm = ChatOpenAI(model="gpt-4o")


#result = llm.invoke(messages)
result2 = llm_hf.invoke(messages)

#print(f"Answer from OpenAI: {result.content}")
print(f"Answer from HuggingFace: {result2.content}")

