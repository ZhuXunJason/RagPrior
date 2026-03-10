"""
Configuration file for the Bug Report Prioritization project.
Loads sensitive information from .env file using python-dotenv.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Neo4j Database Configuration
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
NEO4J_AUTH = (NEO4J_USERNAME, NEO4J_PASSWORD)

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    print("Warning: GOOGLE_API_KEY not found in .env file or environment variables.")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = os.getenv("BASE_URL", "https://api.openai.com/v1")

if not OPENAI_API_KEY:
    print("Warning: OPENAI_API_KEY not found in .env file or environment variables.")

# Project Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESPONSE_DIR = os.path.join(PROJECT_ROOT, "response")
RESULT_DIR = os.path.join(PROJECT_ROOT, "result")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "reports")