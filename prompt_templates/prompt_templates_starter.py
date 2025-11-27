from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatOpenAI(model="gpt-4o")

#template = "Write a {tone} email to {company} expressing interest in the {position} position, mentioning {skill} as a key strength. Keep it to 4 lines max"

#prompt_template = ChatPromptTemplate.from_template(template)

'''prompt = prompt_template.invoke({
    "tone": "energetic",
    "company": "samsung",
    "position": "AI engineer",
    "skill": "AI"
})'''

#print(prompt)
#print(prompt_template)

# Example 2: Prompt with System and Human Messages (Using Tuples)

messages = [
    ("system", "You are a comediant who tells jokes about{topic}."),
    ("human", " Tell me {joke_count} jokes."),
    ]

prompt_template = ChatPromptTemplate.from_messages(messages)
prompt = prompt_template.invoke({
    "topic": " Germany ",
    "joke_count": " 3 "
})

result = llm.invoke(prompt)

print(result.content)

