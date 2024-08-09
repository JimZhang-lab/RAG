import os
from dotenv import load_dotenv
load_dotenv()
from langchain_openai import ChatOpenAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import AIMessage,SystemMessage,HumanMessage
from langchain.chains.llm import LLMChain
from langchain_core.messages import SystemMessage
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import MessagesPlaceholder, HumanMessagePromptTemplate, ChatPromptTemplate

llm_glm = ChatOpenAI(
    model = os.getenv("ZHIPUAI_MODEL_4airx"),
    streaming=True, 
    callbacks=[StreamingStdOutCallbackHandler()],
    base_url= os.getenv("ZHIPUAI_BASE_URL"),
    api_key = os.getenv("ZHIPUAI_API_KEY"),
    )

prompt  =  ChatPromptTemplate.from_messages(
    [SystemMessage(content="你是一个与人类对话的机器人。"),

    MessagesPlaceholder(variable_name="chat_history"),

    HumanMessagePromptTemplate.from_template("{question}")]
)

# 创建ConversationBufferMemory

memory  =   ConversationBufferMemory(memory_key="chat_history", return_messages=True)

chain = LLMChain(llm=llm_glm, prompt=prompt, memory=memory)

res = chain.invoke({"question":"我是小鸡"})
print(str(res)+"\n")

res = chain.invoke({"question":"我是谁?"})
print(str(res)+"\n")
# messages=[
#     SystemMessage(
#         content="your are a helpful assistant"
#     ),
#     HumanMessage(
#         content="小说梗概应该由哪些组成？"
#     ),
#     AIMessage(
#         content="小说的主要内容是什么？"
#     ),
#     HumanMessage(
#         content="小说的主要内容是描述故事的主要情节，主要人物的故事，以及人物之间的关系。"
#     ),
#     AIMessage(
#         content="好的，小说的主要内容是描述故事的主要情节，主要人物的故事，以及人物之间的关系。"
#     ),
#     HumanMessage(
#         content="那小说的结构是什么样的？"
#     ),
#     AIMessage(
#         content="好的，小说的结构是由前言、序章、正文、后记四个部分组成。"
#     ),
#     HumanMessage(
#         content="好的，谢谢！"
#     )
# ]
# print(llm_glm(messages))