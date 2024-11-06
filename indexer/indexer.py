# Create vector index / embeddings
# parser.add_argument("--docs_dir", type=str, default="./docs/", help="Folder containing data to index")
# parser.add_argument("--index_dir", type=str, default="./index/", help="Path to store the serialized VectorStore")

import os
import argparse
from llama_index.core import Settings, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.indices.vector_store import VectorStoreIndex
from llama_index.core.readers import SimpleDirectoryReader
from llama_index.core.storage import StorageContext

# work around for HugginFace FastToekenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def index():
    parser = argparse.ArgumentParser()
    parser.add_argument("--docs_dir", type=str, default="./docs/", help="folder containing data to index")
    parser.add_argument("--index_dir", type=str, default="./index/", help="path to store the serialized vector store")
    parser.add_argument("--rebuild", type=bool, default=True, help="flag to delete and rebuild the index")
    args = parser.parse_args()

    print(f"using data dir {args.docs_dir}")
    print(f"using index path {args.index_dir}")
    print(f"rebuilding the index? {args.rebuild}")

    embed_model=HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    print(f"Embedding: {embed_model.model_name}")
    llm = OpenAILike(
        model="llama3.2-3b-instruct",
        is_chat_model=True,
        api_base="http://localhost:1337/v1",
        api_key="secret",
        temperature=0.7
    )
    
    Settings.llm = llm
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64
    Settings.embed_model = embed_model

    vector_store = None

    if os.path.exists(args.index_dir):
        if (args.rebuild):
            print("rebuilding index")
            if os.path.exists(args.index_dir):
                for root, dirs, files in os.walk(args.index_dir, topdown=False):
                    for name in files:
                        os.remove(os.path.join(root, name))
                    for name in dirs:
                        os.rmdir(os.path.join(root, name))
                os.rmdir(args.index_dir)

                print("reading docs to create index")
                docs = SimpleDirectoryReader(args.docs_dir).load_data()
                vector_store = VectorStoreIndex.from_documents(docs)
                os.mkdir(args.index_dir)
                vector_store.storage_context.persist(persist_dir=args.index_dir)
                #vector_store
                print ("created vector store from new index")
        else:
            print(f"reading vector store from {args.index_dir}")
            storage_context = StorageContext.from_defaults(
                persist_dir=args.index_dir
            )
            vector_store = load_index_from_storage(
                storage_context=storage_context
            )
            print("loaded vector store from existing index")
    else:
        print("reading docs to create index")
        docs = SimpleDirectoryReader(args.docs_dir).load_data()
        vector_store = VectorStoreIndex.from_documents(docs)
        os.mkdir(args.index_dir)
        vector_store.storage_context.persist(persist_dir=args.index_dir)
        #vector_store
        print ("created vector store from new index")
   
    return vector_store, llm

if __name__ == "__main__":
    index()