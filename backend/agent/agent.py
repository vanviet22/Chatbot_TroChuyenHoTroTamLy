import os
import sys
from dotenv import load_dotenv
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from retrivier_tool.retriever_tool import RetrieverTool
from llm_tool.llm_tool import LLMTool
from searchweb_tool.searchweb_tool import SearchWebTool
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from langchain import hub
import torch
from langchain.memory import ConversationBufferMemory
load_dotenv()

class Agent:
    def __init__(self, tokenizer,model =None, llm = None,):
        # Các tool
        self._rag_tool = RetrieverTool(tokenizer = tokenizer, model = model)
        self._llm_tool = LLMTool(tokenizer = tokenizer, model = model)
        self._searchweb = SearchWebTool()
        # LangChain Agent sẽ sử dụng LLM này để lập luận (ReAct)
        self._llm = llm
    def create_agent(self):
        #1 Dùng prompt chuẩn "react-chat" (hỗ trợ chat_history)
        prompt = hub.pull("hwchase17/react-chat")

        #  Khai báo các công cụ mà agent có thể gọi
        tools = [self._rag_tool, self._llm_tool, self._searchweb]

        # Tạo bộ nhớ hội thoại
        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        # Tạo agent theo ReAct pattern
        agent = create_react_agent(
            llm=self._llm,
            tools=tools,
            prompt=prompt,
        )

        agent_executor = AgentExecutor(
            agent=agent,
            tools=tools,
            memory=memory,
            verbose=True,
            handle_parsing_errors=True
        )

        return agent_executor
