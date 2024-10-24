# https://github.com/LinkedInLearning/introduction-to-ai-orchestration-with-langchain-and-llamaindex-3820082/blob/main/Chap02/rag_llamaindex.py

import os
import argparse
from llama_index.core import (
    Settings, load_index_from_storage
)
from llama_index.core.chat_engine import ContextChatEngine
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.storage import StorageContext
from indexer import index

# work around for HugginFace FastToekenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    vector_store, llm = index()
    retriver = VectorIndexRetriever(vector_store)
    query_engine = RetrieverQueryEngine.from_args (
        retriever=retriver
    )

    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriver,
        query_engine=query_engine,
        llm=llm,
        system_prompt="You are my personal assistant and have access to my calendar events, emails, messages and tasks",
        verbose=True
    )

    chat_engine.chat_repl()

if __name__ == "__main__":
    main()