from llama_index.core.chat_engine import ContextChatEngine
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import Settings

def query_pipeline(vector_store):
    retriver = VectorIndexRetriever(vector_store)
    query_engine = RetrieverQueryEngine.from_args (
        retriever=retriver
    )

    chat_engine = ContextChatEngine.from_defaults(
        retriever=retriver,
        query_engine=query_engine,
        llm=Settings.llm,
        system_prompt="You have access to birthdays, the birthday is given in the format month/day/year. Today's date is 11/5/2024. Answer questions on upcoming birthdays.",
        verbose=True
    )

    chat_engine.chat_repl()

