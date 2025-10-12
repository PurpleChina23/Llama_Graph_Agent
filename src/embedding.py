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



def create_and_save_embedding_index(path: str = "src/raw_data", 
                                    emb_model: str = "text-embedding-v3"):
    load_key()
    print(f'''你配置的 API Key 是：{os.environ["EMBEDDING_KEY"][:5]+"*"*5}''')

    print("\n" + "="*60)
    print("Analysing Doc ...")
    print("="*60)


    # Load documents from the 'raw_data' directory
    documents = SimpleDirectoryReader(path).load_data()

    print("\n" + "="*60)
    print("Creating vector index...")
    print("="*60)

    index = VectorStoreIndex.from_documents(
        documents,

        embed_model=OpenAIEmbedding(
            model = emb_model,
            api_key=os.getenv("EMBEDDING_KEY"),
            api_base=os.getenv("OPENAI_API_BASE")  
        ))


    os.makedirs("src/storage", exist_ok=True)
    # Persist
    index.storage_context.persist("src/storage")

    print("索引文件保存到了knowledge_base/test")


def load_embedding_index(path: str = "src/storage"):
    # Load index from storage without recomputing embeddings
    storage_context = StorageContext.from_defaults(persist_dir=path)
    index = load_index_from_storage(storage_context)  # DO NOT pass embed_model
    print("成功从knowledge_base/test路径加载索引")
    return index

