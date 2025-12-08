# Bring in deps
from dotenv import load_dotenv
load_dotenv()


import streamlit as st


from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.utilities import WikipediaAPIWrapper




# App framework
st.title('🦜🔗 YouTube GPT Creator')
prompt = st.text_input('Plug in your prompt here')


# Prompt templates
title_template = PromptTemplate(
    input_variables=['topic'],
    template='write me a youtube video title about {topic}'
)


script_template = PromptTemplate(
    input_variables=['title', 'wikipedia_research'],
    template=(
        'write me a youtube video script based on this title '
        'TITLE: {title} while leveraging this wikipedia reserch:{wikipedia_research} '
    )
)


# Memory
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


# LLM (chat model, LCEL-style)
llm = ChatOpenAI(temperature=0.9)


# Output parser
parser = StrOutputParser()


# Chains using pipe logic (Prompt -> LLM -> String)
title_chain = title_template | llm | parser
script_chain = script_template | llm | parser


# Wikipedia
wiki = WikipediaAPIWrapper()


# Show stuff to the screen if there's a prompt
if prompt:
    # 1) Generate title
    title = title_chain.invoke({"topic": prompt})
    # Save into memory manually (similar effect to LLMChain+memory)
    title_memory.save_context({"topic": prompt}, {"title": title})


    # 2) Wikipedia research
    wiki_research = wiki.run(prompt)


    # 3) Generate script based on title + wiki research
    script = script_chain.invoke(
        {"title": title, "wikipedia_research": wiki_research}
    )
    script_memory.save_context({"title": title}, {"script": script})


    # Display
    st.write(title)
    st.write(script)


    with st.expander('Title History'):
        st.info(title_memory.buffer)


    with st.expander('Script History'):
        st.info(script_memory.buffer)


    with st.expander('Wikipedia Research'):
        st.info(wiki_research)
