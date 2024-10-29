from llama_index.core.query_pipeline import InputComponent
from llama_index.core.query_pipeline import QueryPipeline

def query_pipeline(query_engine):
    input_component = InputComponent()
    chain = [input_component, query_engine]
    query_pipeline = create_query_pipeline(chain)
    return query_pipeline

def create_query_pipeline(chain, verbose=True):
    """
    Creates and returns a QueryPipeline instance configured with the specified chain of components.

    Parameters:
    - chain (list): A list of components to be used in the pipeline. Each component in the list should be an instance of a module that can be used in a QueryPipeline (e.g., LLMs, query engines).
    - verbose (bool): If True, enables verbose output for the pipeline.

    Returns:
    - QueryPipeline: An instance of QueryPipeline configured with the specified chain of components.
    """
    pipeline = QueryPipeline(
        chain=chain,
        verbose=verbose
    )

    return pipeline