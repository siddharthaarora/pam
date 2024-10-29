import os
import sys
from getpass import getpass
from llama_index.core.settings import Settings
from util.utils import setup_llm, setup_embed_model, setup_vector_store
from index.ingest import ingest
from index.index import index
from retrieval import query_pipeline

def pam(input=None):
    COLLECTION_NAME = "XYZ"

    setup_llm()
    setup_embed_model()
    # vector_store = ingest(vector_store)
    # query_engine = index(vector_store)
    # query = query_pipeline(query_engine)

    # query.run(input)

if __name__ == "__main__":
    pam()