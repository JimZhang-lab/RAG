import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains.llm import LLMChain
from langchain.chains.conversation.base import ConversationChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
#加载 .env文件中的环境变量,保存在代码当前路径下
load_dotenv()
from langchain.memory import ConversationBufferWindowMemory,ConversationBufferMemory


prompt = PromptTemplate(
    input_variables=["chat_history", "question"],
    template="""You are a very kindl and friendly AI assistant. You are
    currently having a conversation with a human. Answer the questions
    in a kind and friendly tone with some sense of humor.
    
    chat_history: {chat_history},
    Human: {question}
    AI:"""
)


# llm = ChatOpenAI(
#     model = os.getenv("OPENAI_MODEL"),
#     base_url = os.getenv("OPENAI_API_URL"),
#     api_key=os.getenv("OPENAI_API_KEY"))

llm_glm = ChatOpenAI(
    model = os.getenv("ZHIPUAI_MODEL_4airx"),
    base_url= os.getenv("ZHIPUAI_BASE_URL"),
    api_key = os.getenv("ZHIPUAI_API_KEY")
    )

# memory = ConversationBufferWindowMemory(memory_key="chat_history")
memory = ConversationBufferWindowMemory(memory_key="chat_history")
llm_chain = LLMChain(
    llm=llm_glm,
    memory=memory,
    prompt=prompt
)
# llm_chain = prompt | llm_glm

st.set_page_config(
    page_title="ChatGPT Clone",
    page_icon="🤖",
    layout="wide"
)


st.title("ChatGPT Clone")

# check for messages in session and create if not exists
if "messages" not in st.session_state.keys():
    st.session_state.messages = [
        {"role": "assistant", "content": "Hello there, am ChatGPT clone"}
    ]
    
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        
user_prompt = st.chat_input()

if user_prompt is not None:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.write(user_prompt)

if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Loading..."):
            ai_response = llm_chain.predict(question=user_prompt)
            st.write(ai_response)
    new_ai_message = {"role": "assistant", "content": ai_response}
    st.session_state.messages.append(new_ai_message)