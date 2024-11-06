import os
import sys
from getpass import getpass
from llama_index.core.settings import Settings
from util.utils import setup_llm, setup_embed_model
from indexer.ingest import ingest
from indexer.index import index
from retrieval import query_pipeline

def pam():
    COLLECTION_NAME = "XYZ"

    setup_llm()
    
    setup_embed_model()
    
    nodes = ingest(
        collection_name=COLLECTION_NAME,
        docs_dir="docs/birthdays"
    )

    vector_index = index(nodes, index_dir="docs/index", rebuild_index=False)
    
    query_pipeline.query_pipeline(vector_index)

if __name__ == "__main__":
    pam()