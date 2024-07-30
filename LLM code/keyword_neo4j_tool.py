from __future__ import annotations
from database import Neo4jDatabase
from langchain.chains.base import Chain

from typing import Any, Dict, List

from pydantic import Field
from logger import logger
from langchain.output_parsers import StructuredOutputParser


fulltext_search = """
CALL db.index.fulltext.queryNodes("osteosarcoma", $query) 
YIELD node, score
RETURN coalesce(node.name, node.title) AS result
LIMIT 100
"""

def generate_params(input_str):
    """
    Generate full text parameters using the Lucene syntax for "osteosarcoma" database
    """
    print("Generating parameters with input:", input_str)
    return f'name:"{input_str}"'

class LLMKeywordGraphChain(Chain):
    """Chain for keyword question-answering against an "osteosarcoma" graph."""
    
    graph: Neo4jDatabase = Field(exclude=True)
    input_key: str = "query"  #: :meta private:
    output_key: str = "result"  #: :meta private:

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
        """Extract entities, look up info and answer question."""
        question = inputs[self.input_key]
        params = generate_params(question)
        print("Generated parameters:", params)
        self.callback_manager.text(
            "Keyword query parameters:", end="\n", verbose=self.verbose
        )
        self.callback_manager.text(
            params, color="green", end="\n", verbose=self.verbose
        )
        logger.debug(f"Keyword search params: {params}")
        context = self.graph.query(
            fulltext_search, {'query': params})
        logger.debug(f"Keyword search context: {context}")
        print("Search context:", context)
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
    
    chain = LLMKeywordGraphChain(llm=llm, verbose=True, graph=osteosarcoma_graph,
            max_retries = 1, return_intermediate_steps = True)

    try:
        output = chain.run("What type of information is available for osteosarcoma?")
        print(output)
    except Exception as e:
        print(f"An error occurred: {e}")
        if hasattr(chain, 'intermediate_steps'):
            print("Last thought:")
            print(chain.intermediate_steps[-1].agent_call.full_output)
