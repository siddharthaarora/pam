import os
from llama_index.core import Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.ingestion import IngestionCache, IngestionPipeline
from llama_index.core.storage import StorageContext
from llama_index.core.node_parser.text import SentenceSplitter, TokenTextSplitter
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.readers import SimpleDirectoryReader

# work around for HugginFace FastToekenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def ingest(collection_name, docs_dir="../docs"):
    documents = SimpleDirectoryReader(
        input_dir = docs_dir, 
        filename_as_id=True).load_data()

    ingest_cache = IngestionCache(
        collection="birthdays",
    )

    # create pipeline with transformations
    pipeline = IngestionPipeline(
        transformations=[
            TokenTextSplitter(chunk_size=256, chunk_overlap=16),
            Settings.embed_model
        ],
        docstore=SimpleDocumentStore(),
        cache=ingest_cache,
    )

    # run the pipeline
    nodes = pipeline.run(documents = documents)

    return nodes
