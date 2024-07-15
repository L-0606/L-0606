import os
import logging
from agent import OsteosarcomaAgent
from database import Neo4jDatabase
from fastapi import APIRouter, HTTPException, Query
from run import get_result_and_thought_using_graph
from py2neo import DatabaseError, Graph

host = "bolt://localhost:7687"
user = "neo4j"
password = "12345"
model_name = "ChatGLM3"

# build router
router = APIRouter()
logger = logging.getLogger(__name__)
graph = Neo4jDatabase(host=host, user=user, password=password)
print("Neo4jDatabase object created.")

# 尝试连接数据库并执行简单的查询
try:
    # 使用 Graph 对象的构造函数来连接数据库
    graph.graph = Graph(host, auth=(user, password))

    # 尝试运行简单的查询来验证连接
    result = graph.graph.run("MATCH (n) RETURN count(n) as count").data()
    if result:
        print("Database connected successfully.")
    else:
        print("Failed to connect to the database.")
except Exception as e:
    print("Failed to connect to the database:", e)

agent_osteosarcoma = OsteosarcomaAgent.initialize(osteosarcoma_graph=graph, neo4j_host=host,
                                                  neo4j_user=user, neo4j_password=password,
                                                  model_name=model_name)
print("OsteosarcomaAgent object initialized.")

# 检查代理是否加载成功
if agent_osteosarcoma:
    print("Osteosarcoma agent loaded successfully.")
else:
    print("Failed to load the osteosarcoma agent.")

@router.get("/predict")
def get_load(message: str = Query(...)):
    try:
        print("Received message:", message)
        result = get_result_and_thought_using_graph(agent_osteosarcoma, graph, message)
        print("Result:", result)
        return result
    except Exception as e:
        # Log stack trace
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e)) from e
