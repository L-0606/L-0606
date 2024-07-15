from langchain.agents.agent import AgentExecutor
from langchain.agents.tools import Tool
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
from langchain.agents import initialize_agent
from langchain.agents import AgentType
from langchain.agents.conversational_chat.base import ConversationalChatAgent
from cypher_database_tool import LLMCypherGraphChain
from keyword_neo4j_tool import LLMKeywordGraphChain
from vector_neo4j_tool import LLMNeo4jVectorChain
import os

'''
You are an osteosarcoma expert, and you can answer all your users' questions based on your knowledge in the graph database,When you answer the question, only use one language reply, do not mix Chinese and English.
'''

os.environ["no_proxy"] = '127.0.0.1,localhost'

class OsteosarcomaAgent(AgentExecutor):
    """Osteosarcoma agent"""

    @staticmethod
    def function_name():
        return "OsteosarcomaAgent"

    @classmethod
    def initialize(cls, osteosarcoma_graph, neo4j_host,
                   neo4j_user, neo4j_password, *args, **kwargs):
        llm = ChatOpenAI(
            model_name="ChatGLM3",
            # model_path="/dssg/home/acct-medhyq/medhyq-zll/chatGLM3/chatglm3-6b",
            openai_api_base="http://localhost:8000/v1",
            openai_api_key="EMPTY",
            streaming=True
        )

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        readonlymemory = ReadOnlySharedMemory(memory=memory)
        cypher_tool = LLMCypherGraphChain(
            llm=llm, graph=osteosarcoma_graph, verbose=True, memory=readonlymemory)
        fulltext_tool = LLMKeywordGraphChain(
            llm=llm, graph=osteosarcoma_graph, verbose=True)
        vector_tool = LLMNeo4jVectorChain(
            llm=llm, verbose=True, graph=osteosarcoma_graph
        )

        # Load the tool configs that are needed.
        tools = [
            Tool(
                name="Cypher search",
                func=cypher_tool.run,
                description="""
                Utilize this tool to search within a osteosarcoma database, specifically designed to answer osteosarcoma-related questions.
                This specialized tool offers streamlined search capabilities to help you find the osteosarcoma information you need with ease.
                Input should be full question.""",
            ),
            Tool(
                name="Keyword search",
                func=fulltext_tool.run,
                description="""
                Use this tool when explicitly told to search by keyword.
                The input should be a list of relevant osteosarcoma information inferred from the question.
                Remove the stop word "This" from the specified osteosarcoma information. """,
            ),
            Tool(
                name="Vector search",
                func=vector_tool.run,
                description="Utilize this tool when explicity told to use vector search.Input should be full question.Do not include agent instructions.",
            ),
        ]
        agent=ConversationalChatAgent.from_llm_and_tools(
            llm=llm,
            tools=tools,
            outparser="Make up to three thought.'"

        )	
        def _handle_error(error) -> str:
            return str(error)[:50]

        agent_chain = initialize_agent(tools, llm, verbose=True,
                                       memory=memory, handle_parsing_errors=True, return_direct=True, max_retries=1
                                       )
        # print(agent_chain)
        return agent_chain

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)
