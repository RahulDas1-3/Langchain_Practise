from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI

# Load environment variables from .env
load_dotenv()

# Create a ChatOpenAI model
model = ChatOpenAI(model="gpt-4o")

# Define prompt templates
planet_facts_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You like telling facts and you tell facts about {planets}."),
        ("human", "Tell me {count} facts."),
    ]
)

# Define a prompt template for translation to French
translation_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a translator and convert the provided text into {language}."),
        ("human", "Translate the following text to {language}: {text}"),
    ]
)

# Define additional processing steps using RunnableLambda
count_words = RunnableLambda(lambda x: f"\n{x}\nWord count: {len(x.split())}")
prepare_for_translation = RunnableLambda(lambda output: {"text": output, "language": "bengali"})


# Create the combined chain using LangChain Expression Language (LCEL)
chain = planet_facts_template | model | StrOutputParser() | prepare_for_translation | translation_template | model | StrOutputParser() | count_words 

# Run the chain
result = chain.invoke({"planets": "Mars", "count": 2})

# Output
print(result)