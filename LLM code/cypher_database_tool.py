from database import Neo4jDatabase
from pydantic import BaseModel, Extra
from langchain.prompts.base import BasePromptTemplate
from langchain.chains.llm import LLMChain
from langchain.chains.base import Chain
from langchain.memory import ReadOnlySharedMemory,ConversationBufferMemory
from langchain.prompts import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import Dict, List, Any

from logger import logger

SYSTEM_TEMPLATE = """
You are an assistant that can generate Cypher queries based on the provided examples.
Here are some example Cypher queries:
"""
examples = """
# What is osteosarcoma?
MATCH (m: disease {Name: "osteosarcoma"})-[r:define]->(a)
RETURN {a.Name} AS result

# What genes promote the growth of osteosarcoma?
MATCH (osteosarcoma:disease {Name: "osteosarcoma"})<-[r:promote]-(a)
RETURN {gene: a.Name} AS result

# What are the characteristics of osteosarcoma?
MATCH (m:disease {Name: "osteosarcoma"})-[r:possess]->(a)
RETURN {characteristic: a.Name} AS result

# What genes regulate CDK2?
MATCH (m:protein {Name: "CDK2"})<-[r:regulate]-(n)
RETURN {gene: n.Name} AS result

# Do you know what are the characteristics of primary osteosarcoma?
MATCH (m:classification {Name: "primary osteosarcoma"})-[r:possess]->(a)
RETURN {characteristic: a.Name} AS result

# What are the mirnas associated with the development of osteosarcoma?
MATCH (m:RNA)-[:correlation]->(:characteristic {Name:"development of osteosarcoma"})
RETURN {mirna: m.Name} AS result
ORDER BY SIMILARITY DESC LIMIT 5

Please provide the Cypher query that best answers the user's question. Feel free to provide any additional context or explanation that you think might be helpful.
"""

SYSTEM_CYPHER_PROMPT = SystemMessagePromptTemplate.from_template(
    SYSTEM_TEMPLATE)
HUMAN_TEMPLATE = "{question}"
HUMAN_PROMPT = HumanMessagePromptTemplate.from_template(HUMAN_TEMPLATE)


class LLMCypherGraphChain(Chain):
    """Chain that interprets a prompt and executes python code to do math.
    """

    llm: Any
    """LLM wrapper to use."""
    system_prompt: BasePromptTemplate = SYSTEM_CYPHER_PROMPT
    human_prompt: BasePromptTemplate = HUMAN_PROMPT
    input_key: str = "question"  #: :meta private:
    output_key: str = "answer"  #: :meta private:
    graph: Neo4jDatabase
    memory: ReadOnlySharedMemory = None  


    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid
        arbitrary_types_allowed = True

    @property
    def input_keys(self) -> List[str]:
        """Expect input key.
        :meta private:
        """
        return [self.input_key]

    @property
    def output_keys(self) -> List[str]:
        """Expect output key.
        :meta private:
        """
        return [self.output_key]

    def _call(self, inputs: Dict[str, str]) -> Dict[str, str]:
        logger.debug(f"Cypher generator inputs: {inputs}")
        chat_prompt = ChatPromptTemplate.from_messages(
            [self.system_prompt] + inputs['chat_history']+  [self.human_prompt])
        cypher_executor = LLMChain(prompt=chat_prompt, llm=self.llm, 
            callback_manager=self.callback_manager)
        cypher_statement = cypher_executor.predict(
            question=inputs[self.input_key], stop=["Output:"])

        print("Cypher Statement:", cypher_statement)

        self.callback_manager.text(
            "Generated Cypher statement:", color="green", end="\n", verbose=self.verbose
        )
        self.callback_manager.text(
            #cypher_statement, color="blue", end="\n", verbose=self.verbose
        )
         #If Cypher statement was not generated due to lack of context
        if not "MATCH" in cypher_statement:
            print("Missing context to create a Cypher statement")
            return {'answer': 'Missing context to create a Cypher statement'}
        context = self.graph.query(cypher_statement)

        print("Context:", context)
        logger.debug(f"Cypher generator context: {context}")

        return {'answer': context}

if __name__ == "__main__":

    from langchain.chat_models import ChatOpenAI
    import os
    
    os.environ["no_proxy"]='127.0.0.1,localhost'
    llm = ChatOpenAI(
        model_name="ChatGLM3",
        openai_api_base="http://localhost:8000/v1",
        openai_api_key="EMPTY",
        temperature=0.8,
        max_tokens=999999,
      
        streaming=True
    )
    
    osteosarcoma_graph = Neo4jDatabase(host="bolt://lacolhost:7687",
                             user="neo4j", password="12345")
    memory = ConversationBufferMemory(
        memory_key="chat_history", return_messages=True)
    readonlymemory = ReadOnlySharedMemory(memory=memory)

    chain = LLMCypherGraphChain(llm=llm, verbose=True, graph=osteosarcoma_graph,memory=readonlymemory)

    #try:
    output = chain.run("What genes promote the growth of osteosarcoma?")
    print(output)
    #except Exception as e:
        #print(f"An error occurred: {e}")
        #if hasattr(chain, 'intermediate_steps'):
            #print("Last thought:")
            #print(chain.intermediate_steps[-1].agent_call.full_output)

