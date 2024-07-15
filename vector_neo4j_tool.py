from __future__ import annotations
from database import Neo4jDatabase
#from langchain.embeddings import OpenAIEmbeddings
#from embeddings import ChatGLM3Embeddings
from langchain.embeddings.huggingface import HuggingFaceInstructEmbeddings
#from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains.base import Chain

from typing import Any, Dict, List

from pydantic import Field
from logger import logger
from langchain.output_parsers import StructuredOutputParser

vector_search = """
WITH $embedding AS e
CALL db.index.vector.queryNodes('osteosarcoma', 5, e) YIELD node AS m, score
CALL {
  WITH m
  MATCH (m)-[r:!RATED]->(target)
  WHERE target:Osteosarcoma  // Match only nodes related to osteosarcoma
  RETURN coalesce(m.name, m.title) + " " + type(r) + " " + coalesce(target.name) AS result
  UNION
  WITH m
  MATCH (m)<-[r:!RATED]-(target)
  WHERE target:Osteosarcoma  // Match only nodes related to osteosarcoma
  RETURN coalesce(target.name, target.title) + " " + type(r) + " " + coalesce(m.name, m.title) AS result
}
RETURN result LIMIT 100
"""

class LLMNeo4jVectorChain(Chain):
    """Chain for question-answering against a graph."""
    
    graph: Neo4jDatabase = Field(exclude=True)
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

    embeddings: embedding = HuggingFaceInstructEmbeddings(model_name='instructor-xl') # 使用正确的类型

    @property
    def input_keys(self) -> List[str]:
        """Return the input keys.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Return the output keys.
        :meta private:
        """
        _output_keys = [self.output_key]
        return _output_keys

    def _call(self, inputs: Dict[str, str]) -> Dict[str, Any]:
        """Embed a question and do vector search."""
        question = inputs[self.input_key]
        logger.debug(f"Vector search input: {question}")
        embedding = self.embeddings.embed_query(inputs, question)  # 将inputs和question传递给embed_query

        print("Embedding:", embedding)  # 添加打印语句
        self.callback_manager.text(
            "Vector search embeddings:", end="\n", verbose=self.verbose)
        self.callback_manager.text(
            embedding[:5], color="green", end="\n", verbose=self.verbose)
        context = self.graph.query(
            vector_search, {'embedding': embedding})

        print("Context:", context)  # 添加打印语句
        return {self.output_key: context}

if __name__ == '__main__':
    from langchain.chat_models import ChatOpenAI
    import os 
    
    os.environ["no_proxy"]='127.0.0.1,localhost'

    llm = ChatOpenAI(
        model_name="ChatGLM3",
	openai_api_base="http://localhost:8000/v1",
	openai_api_key="EMPTY",
        temperature= 0.8,
        max_tokens= 999999,
        top_p = 0.8,
        top_k = 2,
        streaming=True
    )
    osteosarcoma_graph = Neo4jDatabase(host="bolt://localhost:7687",
                             user="neo4j", password="12345")
    chain = LLMNeo4jVectorChain(llm=llm, verbose=True, graph=osteosarcoma_graph, 
            max_retries = 1, return_intermediate_steps = True)

    try:
        output = chain.run("What RNA is associated with osteosarcoma?")
        print(output)
    except Exception as e:
        print(f"An error occurred: {e}")
        if hasattr(chain, 'intermediate_steps'):
            print("Last thought:")
            print(chain.intermediate_steps[-1].agent_call.full_output)
