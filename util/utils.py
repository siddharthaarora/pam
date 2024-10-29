import random
import time
from tqdm import tqdm
from collections import defaultdict
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.ingestion import IngestionPipeline
from llama_index.core.settings import Settings
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike

def setup_llm(provider="local", model=None, api_key=None, **kwargs):
    """
    Configures the LLM (Language Learning Model) settings.

    Parameters:
    - api_key (str): The API key for authenticating with the LLM service.
    - llm_model (str): The model identifier for the LLM service.
    """
    llm = None
    if provider == "openai":
        llm = OpenAI(model=model, api_key=api_key, **kwargs)
    elif provider == "local":
        llm = OpenAILike(
            model="llama3.2-3b-instruct",
            is_chat_model=True,
            api_base="http://localhost:1337/v1",
            api_key="secret",
            temperature=0.7
        )
    else:
        raise ValueError(f"Invalid provider: {provider}. Pick one of 'cohere', 'openai', or 'mistral'.")

    Settings.llm = llm
    Settings.chunk_size = 512
    Settings.chunk_overlap = 64

def setup_embed_model(provider="fastembed", **kwargs):
    """
    Configures the embedding model settings.

    Parameters:
    - model_name (str): The model identifier for the embedding service.
    """
    if provider == "openai":
        Settings.embed_model = OpenAIEmbedding(model_name="text-embedding-3-large", **kwargs)
    elif provider == "fastembed":
        Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-base-en-v1.5", **kwargs)
    else:
        raise ValueError(f"Invalid provider: {provider}. Pick one of 'cohere', 'fastembed', or 'openai'.")

# def group_documents_by_author(documents):
#     """
#     Group documents by author.

#     This function organizes a list of document objects into a dictionary where each key is an author's name,
#     and the value is a list of all documents authored by that person. It leverages defaultdict to automatically
#     handle any authors not previously encountered without raising a KeyError.

#     Args:
#         documents (list): A list of document objects, each having a 'metadata' dictionary that includes an 'author' key.

#     Returns:
#         defaultdict: A dictionary where keys are author names and values are lists of documents for those authors.
#     """
#     # Initialize a defaultdict to store lists of documents, organized by author.
#     documents_by_author = defaultdict(list)

#     # Loop over each document in the provided list.
#     for doc in documents:
#         # Retrieve the 'author' from the document's metadata. Default to None if 'author' key is missing.
#         author = doc.metadata.get('author', None)

#         # Check if the author exists. If so, append the document to the corresponding list in the dictionary.
#         if author:
#             documents_by_author[author].append(doc)
#         else:
#             # If no author is specified, print a warning. These documents will not be added to any author group.
#             print("Warning: A document without an author was encountered and skipped.")

#     # Return the populated defaultdict containing grouped documents.
#     return documents_by_author

# def sample_documents(documents_by_author, num_samples=10):
#     """
#     Randomly sample a specific number of documents for each author from a grouped dictionary.
#     Only documents with more than 500 characters are considered for sampling.

#     This function takes a dictionary where each key is an author's name and the value is a list of document
#     objects authored by that person. It attempts to sample a specified number of documents for each author.
#     If an author does not have enough documents to meet the sample size, it prints a warning.

#     Args:
#         documents_by_author (dict): A dictionary where keys are authors' names and values are lists of documents.
#         num_samples (int): The desired number of documents to sample from each author's list.

#     Returns:
#         list: A list containing the randomly sampled documents across all authors, up to the specified number
#               per author, where possible.
#     """
#     # Initialize an empty list to store the sampled documents.
#     sampled_documents = []

#     # Iterate over each author and their corresponding documents in the dictionary.
#     for author, docs in documents_by_author.items():
#         # Filter documents with more than 500 characters.
#         valid_docs = [doc for doc in docs if len(doc.get_content()) > 500]

#         # Check if the current author has enough documents to meet the requested sample size.
#         if len(valid_docs) >= num_samples:
#             # If yes, randomly sample the documents and extend the sampled_documents list with the results.
#             sampled_documents.extend(random.sample(valid_docs, num_samples))
#         else:
#             # If no, print a warning message indicating the author and the deficiency in document count.
#             print(f"Author {author} does not have enough valid documents to sample {num_samples}.")

#     # Return the list of all sampled documents.
#     return sampled_documents

# def run_generations_on_eval_set(eval_dataset, col_name, query_pipeline, time_out=True):
#     """
#     Processes an evaluation dataset to add a new column with answers generated by a query pipeline.
    
#     This function iterates over each item in a provided dataset, uses a query pipeline to generate an answer
#     for each question, and then appends these answers as a new column in the dataset. If `time_out` is True,
#     the function pauses for 25 seconds after every 5 queries to comply with API rate limits. Progress is visually
#     tracked using a tqdm progress bar.
    
#     Parameters:
#         eval_dataset (Dataset): A Hugging Face `Dataset` object containing at least a 'question' field.
#         col_name (str): The name of the new column to add to the dataset with the generated answers.
#         query_pipeline: The query pipeline object used to generate answers.
#         time_out (bool): If True, pauses execution to respect API rate limits. Default is True.
        
#     Returns:
#         Dataset: The original dataset augmented with a new column containing the generated answers.
    
#     Example:
#         from datasets import load_dataset
#         query_pipeline = setup_query_pipeline(api_key="your_api_key")
#         eval_dataset = load_dataset('squad', split='validation').select(range(100))  # Example dataset
#         updated_dataset = run_generations_on_eval_set(eval_dataset, 'generated_answers', query_pipeline)
#         print(updated_dataset)
#     """
#     responses = []

#     # Use tqdm to create a progress bar over the dataset iteration
#     for i, row in tqdm(enumerate(eval_dataset), total=len(eval_dataset), desc="Generating answers"):
#         response = query_pipeline.run(row['question'])
#         responses.append(response.response)
        
#         # Pause after every 5 queries to respect the API rate limit, if time_out is True
#         if time_out and (i + 1) % 5 == 0:
#             time.sleep(25)
    
#     # Add the responses as a new column to the dataset
#     eval_dataset = eval_dataset.add_column(col_name, responses)
#     return eval_dataset
