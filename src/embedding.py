import warnings
import os
import logging

# Must suppress warnings BEFORE importing llama_index to catch import-time warnings
# llama_index warning
warnings.filterwarnings('ignore', category=UserWarning)
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
logging.basicConfig(level=logging.ERROR)

from config.load_key import load_key
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import SimpleDirectoryReader,VectorStoreIndex
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import StorageContext,load_index_from_storage
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler





def create_and_save_embedding_index(load_path: str = "src/raw_data", 
                                    store_path: str = "src/storage"):
    load_key()
    print(f'''你配置的 API Key 是：{os.environ["EMBEDDING_KEY"][:5]+"*"*5}''')

    print("\n" + "="*60)
    print("Analysing Doc ...")
    print("="*60)


    # Load documents from the 'raw_data' directory
    documents = SimpleDirectoryReader(load_path).load_data()

    print("\n" + "="*60)
    print("Creating vector index...")
    print("="*60)

    index = VectorStoreIndex.from_documents(
        documents,

        embed_model=OpenAIEmbedding(
            model = "text-embedding-3-large",
            api_key=os.getenv("EMBEDDING_KEY"),
            api_base=os.getenv("OPENAI_API_BASE")  
        ))


    os.makedirs(store_path, exist_ok=True)
    # Persist
    index.storage_context.persist(store_path)

    print("索引文件保存到了knowledge_base/test")


def load_embedding_index(path: str = "src/storage"):
    # Load index from storage without recomputing embeddings
    #？？？？？？？？？要不要用embedding model api 拿参数？？？？？？？？？？？？
    storage_context = StorageContext.from_defaults(persist_dir=path)
    index = load_index_from_storage(storage_context,
                                    embed_model=OpenAIEmbedding(
                                        model = "text-embedding-3-small",
                                        api_key=os.getenv("EMBEDDING_KEY"),
                                        api_base=os.getenv("OPENAI_API_BASE")))
    
    print("\n" + "="*60)
    print("📦 Index successfully unpacked from knowledge_base")
    print("="*60)

    return index
# ✅ 彻底关闭全局默认 LLM

def read_and_query(user_query: str = "what do we have?"):
    
    load_key()
    index = load_embedding_index()
    query_engine = index.as_query_engine(
        streaming=False,
        llm=OpenAILike(
            model="gpt-5-nano",
            api_base=os.getenv("OPENAI_API_BASE"), 
            api_key = os.getenv("OPENAI_API_KEY"),
            is_chat_model=True
            
            ))
    
    print("\n" + "="*60)
    print("Dont BB I am Thinking ...")
    print("="*60)

    response = query_engine.query(user_query)
    return response


create_and_save_embedding_index(load_path = "src/raw_data/fitting_book",
                                store_path = "src/storage/fitting_book_emb")




