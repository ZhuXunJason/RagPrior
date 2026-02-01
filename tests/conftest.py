# tests/conftest.py
import pytest
from neo4j import GraphDatabase

# 配置你的 Neo4j 地址和密码
URI = "neo4j+s://b67ac1a1.databases.neo4j.io"
AUTH = ("neo4j", "jgHSRjaI99nVVpWpJb2O-Gtzs6l5dJwg5fMwKMhnw3M")

@pytest.fixture(scope="session")
def neo4j_driver():
    """创建一个全局的数据库驱动"""
    driver = GraphDatabase.driver(URI, auth=AUTH)
    yield driver
    driver.close()

@pytest.fixture(scope="function")
def session(neo4j_driver):
    """为每个测试函数提供一个会话"""
    with neo4j_driver.session() as session:
        yield session