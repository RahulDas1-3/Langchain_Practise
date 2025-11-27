from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a facts expert who knows facts about {planets}. "),
        ("human", " Tell me a {fact_count} facts."),
    ]
)

chain = prompt_template | llm | StrOutputParser()

result = chain.invoke({"planets": "Neptune", "fact_count": "3"})

print(result)
