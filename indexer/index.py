import os
from llama_index.core import Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.storage import StorageContext
from llama_index.core.node_parser.text import SentenceSplitter, TokenTextSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore

def index(nodes, index_dir="../docs/index", rebuild_index=False):

    vector_store = VectorStoreIndex(nodes)

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
        )

    # vector_index = create_index(
    #     from_where="vector_store",
    #     embed_model=Settings.embed_model, 
    #     vector_store=vector_store, 
    #     )
    
    vector_store.storage_context.persist(persist_dir=index_dir)

    return vector_store

def create_index(from_where, **kwargs):
    """
    Creates and returns a VectorStoreIndex instance configured with the specified parameters.

    Parameters:
    **kwargs: Additional keyword arguments for configuring the index, such as:
        - embed_model: The embedding model to be used in the index.
        - vector_store: The vector store to be used in the index.
        - nodes: The nodes to be used in the index.
        - storage_context: The storage context to be used in the index.

    Returns:
    - VectorStoreIndex: An instance of VectorStoreIndex configured with the specified Qdrant client and vector store.
    """
    if from_where=="vector_store":
        index = VectorStoreIndex.from_vector_store(embed_model=Settings.embed_model, **kwargs)
        return index
    elif from_where=="docs":
        index = VectorStoreIndex.from_documents(embed_model=Settings.embed_model, **kwargs)
        return index
    else:
        raise ValueError(f"Invalid option: {from_where}. Pick one of 'vector_store', or 'docs'.")

def create_query_engine(index, mode, **kwargs):
    """
    Creates and returns a query engine from the given index with the specified configurations.

    Parameters:
    - index: The index object from which to create the query engine. This should be an instance of VectorStoreIndex or similar, which has the as_query_engine method.
    - mode (str): The mode of the query engine to create. Possible values are "chat", "query", and "retrieve".
    - **kwargs: Additional keyword arguments for configuring the query engine, such as similarity_top_k and return_sources.

    Returns:
    - A query engine configured with the specified parameters.
    """
    if mode =="chat":
        return index.as_chat_engine(**kwargs)

    elif mode == "query":
        return index.as_query_engine(**kwargs)

    elif mode == "retrieve":
        return index.as_retriever(**kwargs)
    else:
        raise ValueError(f"Invalid mode: {mode}. Pick one of 'chat', 'query', or 'retrieve'.")
