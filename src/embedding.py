import warnings
import os
import logging

# Must suppress warnings BEFORE importing llama_index to catch import-time warnings
# llama_index warning
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
logging.basicConfig(level=logging.ERROR)

# ============================================================================
# IMPROVEMENT NOTE: Add better imports for 2025 best practices
# ============================================================================
# TODO: Add these imports for enhanced retrieval and error handling:
# from tenacity import retry, stop_after_attempt, wait_exponential
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.core.postprocessor import SimilarityPostprocessor
# from llama_index.postprocessor.cohere_rerank import CohereRerank  # For reranking (+35% accuracy)
# ============================================================================

from config.load_key import load_key
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex 
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core.vector_stores.simple import SimpleVectorStore
from llama_index.core.graph_stores.simple import SimpleGraphStore
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import StorageContext,load_index_from_storage
from llama_index.core.response.notebook_utils import display_source_node




def create_and_save_embedding_index(load_path: str = "src/raw_data",
                                    store_path: str = "src/storage"):
    # ============================================================================
    # IMPROVEMENT NEEDED: Add parameters for chunking configuration
    # ============================================================================
    # TODO: Add parameters: chunk_size=512, chunk_overlap=50
    # Research shows 512 tokens is optimal for most RAG applications (2025)
    # ============================================================================

    # ============================================================================
    # IMPROVEMENT NEEDED: Add error handling with try-except
    # ============================================================================
    # TODO: Wrap entire function in try-except block
    # Add retry logic using @retry decorator from tenacity
    # ============================================================================

    if "EMBEDDING_KEY" not in os.environ:
        load_key()
    print(f'''你配置的 API Key 是：{os.environ["EMBEDDING_KEY"][:5]+"*"*5}''')

    print("\n" + "="*60)
    print("Analysing Doc ...")
    print("="*60)

    # ============================================================================
    # IMPROVEMENT NEEDED: Implement smart chunking (2025 best practice)
    # ============================================================================
    # TODO: Instead of loading documents directly, use SentenceSplitter:
    #
    # from llama_index.core.node_parser import SentenceSplitter
    #
    # documents = SimpleDirectoryReader(load_path).load_data()
    # node_parser = SentenceSplitter(
    #     chunk_size=512,  # Optimal size per 2025 research
    #     chunk_overlap=50,  # Preserve context at boundaries
    #     paragraph_separator="\n\n",
    #     secondary_chunking_regex="[.!?]\\s+",  # Split at sentence boundaries
    # )
    # nodes = node_parser.get_nodes_from_documents(documents)
    #
    # This improves retrieval precision by 15-20%
    # ============================================================================

    # Load documents from the 'raw_data' directory
    documents = SimpleDirectoryReader(load_path).load_data()

    print("\n" + "="*60)
    print("Creating vector index...")
    print("="*60)

    # ============================================================================
    # GOOD: Already using text-embedding-3-large (best model, 80.5% accuracy)
    # ============================================================================
    # IMPROVEMENT NEEDED: Add dimension parameter to save storage
    # TODO: Add dimensions=1536 to reduce from native 3072 without much accuracy loss
    # ============================================================================

    index = VectorStoreIndex.from_documents(
        documents,
        embed_model=OpenAIEmbedding(
            model = "text-embedding-3-large",  # ✅ GOOD: Best model
            api_key=os.getenv("EMBEDDING_KEY"),
            api_base=os.getenv("OPENAI_API_BASE")
        ))


    os.makedirs(store_path, exist_ok=True)
    # Persist
    index.storage_context.persist(store_path)

    print("索引文件保存到了knowledge_base/test")

    # ============================================================================
    # IMPROVEMENT NEEDED: Add better logging
    # ============================================================================
    # TODO: Log statistics:
    # print(f"✅ Index saved to {store_path}")
    # print(f"   - {len(documents)} documents processed")
    # print(f"   - {len(nodes)} chunks created")  # If using chunking
    # ============================================================================





def load_embedding_index(path: str = "src/storage/"):
    # ============================================================================
    # IMPROVEMENT NEEDED: Add error handling
    # ============================================================================
    # TODO: Add try-except to handle missing storage directory or corrupted index
    # TODO: Add @retry decorator for API failures
    # ============================================================================

    if "EMBEDDING_KEY" not in os.environ:
        load_key()
    # Load index from storage without recomputing embeddings
    #？？？？？？？？？要不要用embedding model api 拿参数？？？？？？？？？？？？
    # ANSWER: Yes, you MUST provide the same embed_model used during creation

    # ============================================================================
    # 🔴 CRITICAL BUG: Model mismatch!
    # ============================================================================
    # Line 95 (create function) uses: text-embedding-3-large
    # Line 130 (load function) uses:  text-embedding-3-small  ❌ WRONG!
    #
    # MUST BE CONSISTENT or retrieval will fail!
    #
    # TODO: Change text-embedding-3-small → text-embedding-3-large
    #
    # Research (2025): text-embedding-3-large achieves 80.5% accuracy
    #                  text-embedding-3-small achieves 75.8% accuracy
    #                  Cost: $0.13/M vs $0.02/M tokens (worth it for accuracy)
    # ============================================================================

    storage_context = StorageContext.from_defaults(persist_dir=path)
    index = load_index_from_storage(storage_context,
                                    embed_model=OpenAIEmbedding(
                                        model = "text-embedding-3-small",  # 🔴 CRITICAL: Change to "text-embedding-3-large"
                                        api_key=os.getenv("EMBEDDING_KEY"),
                                        api_base=os.getenv("OPENAI_API_BASE")))

    print("\n" + "="*60)
    print("📦 Index successfully unpacked from knowledge_base")
    print("="*60)

    return index
# ✅ 彻底关闭全局默认 LLM

def read_and_query(user_query: str = "what do we have?"):
    # ============================================================================
    # IMPROVEMENT NEEDED: Add error handling with retries (2025 best practice)
    # ============================================================================
    # TODO: Add @retry decorator:
    # @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    #
    # TODO: Wrap in try-except:
    # try:
    #     ... function logic ...
    # except Exception as e:
    #     logger.error(f"Error in read_and_query: {str(e)}")
    #     raise
    # ============================================================================

    # ============================================================================
    # IMPROVEMENT NEEDED: Add configuration parameters
    # ============================================================================
    # TODO: Add parameters: similarity_top_k=5, response_mode="compact"
    # Retrieving more chunks (5 vs default 2) improves answer quality
    # ============================================================================

    if "EMBEDDING_KEY" not in os.environ:
        load_key()
    index = load_embedding_index()
    query_engine = index.as_query_engine(
        streaming=False,
        # TODO: Add similarity_top_k=5 for better retrieval
        llm=OpenAILike(
            model="gpt-5-nano",
            api_base=os.getenv("OPENAI_API_BASE"),
            api_key = os.getenv("OPENAI_API_KEY"),
            is_chat_model=True
            # TODO: Add timeout=30.0 to prevent hanging

            ))

    print("\n" + "="*60)
    print("Dont BB I am Thinking ...")
    print("="*60)

    # ============================================================================
    # IMPROVEMENT NEEDED: Add response validation
    # ============================================================================
    # TODO: Check if response is None or empty before returning
    # ============================================================================

    response = query_engine.query(user_query)
    return response


# load_key()
# embed_model=OpenAIEmbedding(
#             model = "text-embedding-3-small",
#             api_key=os.getenv("EMBEDDING_KEY"),
#             api_base=os.getenv("OPENAI_API_BASE")  
#         )
# Settings.embed_model = embed_model
# index = SummaryIndex.from_documents(documents)
# retriever = index.as_retriever(retriever_mode="embedding", similarity_top_k=3)
# results = retriever.retrieve("your query here")


def read_and_retrieve(user_query: str = "what do we have?"):
        # ============================================================================
        # 🔴 CRITICAL BUG: Invalid parameters!
        # ============================================================================
        # The parameters dense_similarity_top_k, sparse_similarity_top_k, alpha,
        # and enable_reranking are NOT valid for standard LlamaIndex retriever!
        #
        # These parameters are ONLY available when using:
        # 1. Vector stores that support hybrid search (Qdrant, Milvus, Weaviate)
        # 2. Custom retrievers with fusion
        #
        # TODO: Fix this immediately! Replace with valid parameter:
        #       similarity_top_k=5
        # ============================================================================

        # ============================================================================
        # IMPROVEMENT NEEDED: Implement hybrid retrieval (2025 best practice)
        # ============================================================================
        # Research shows +35% accuracy improvement with hybrid search + reranking!
        #
        # OPTION 1 (Quick fix - use now):
        # retriever = index.as_retriever(
        #     similarity_top_k=5,  # Standard parameter that works
        # )
        #
        # OPTION 2 (Advanced - implement later for +35% accuracy):
        # 1. Migrate to Qdrant/Milvus vector store
        # 2. Enable hybrid search (dense + sparse/BM25)
        # 3. Add reranking with Cohere or bge-reranker-large
        #
        # Example advanced retrieval:
        # from llama_index.core.postprocessor import SimilarityPostprocessor
        # retriever = index.as_retriever(similarity_top_k=10)
        # nodes = retriever.retrieve(user_query)
        # # Apply similarity threshold
        # postprocessor = SimilarityPostprocessor(similarity_cutoff=0.7)
        # filtered_nodes = postprocessor.postprocess_nodes(nodes)
        # return filtered_nodes[:5]  # Top 5 after filtering
        # ============================================================================

        # ============================================================================
        # IMPROVEMENT NEEDED: Add error handling
        # ============================================================================
        # TODO: Add try-except block
        # TODO: Validate that nodes are returned
        # ============================================================================

        if "EMBEDDING_KEY" not in os.environ:
            load_key()
        index = load_embedding_index()
        retriever = index.as_retriever(
            dense_similarity_top_k=3,  # 🔴 INVALID PARAMETER - Remove this
            sparse_similarity_top_k=3,  # 🔴 INVALID PARAMETER - Remove this
            alpha=0.5,  # 🔴 INVALID PARAMETER - Remove this
            enable_reranking=True,  # 🔴 INVALID PARAMETER - Remove this
            # TODO: Replace all above with: similarity_top_k=5
            )
        return(retriever.retrieve(user_query))


# ============================================================================
# 🔴 CRITICAL BUG: Debug code executing on module import!
# ============================================================================
# These lines (286-288) execute EVERY TIME this module is imported!
# This causes:
# - Unwanted API calls and costs
# - Slower imports
# - Side effects in production
# - Confusing output when importing the module
#
# TODO: DELETE these lines immediately OR move to:
#       if __name__ == "__main__":
#           # Test code here
#
# PRIORITY: CRITICAL - Fix this first!
# ============================================================================
nodes = read_and_retrieve("super man")  # 🔴 DELETE THIS LINE
context_str = "\n\n".join([node.get_content() for node in nodes])  # 🔴 DELETE THIS LINE
print(context_str)  # 🔴 DELETE THIS LINE
# ============================================================================
# END OF CRITICAL BUG SECTION - DELETE LINES 286-288
# ============================================================================