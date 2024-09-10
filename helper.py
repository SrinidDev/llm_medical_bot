import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
# from langchain.vectorstores.cassandra import Cassandra
from langchain_community.vectorstores import Cassandra
from langchain_community.llms import Ollama
from cassandra.auth import PlainTextAuthProvider
import tempfile
import cassio
from PyPDF2 import PdfReader
from cassandra.cluster import Cluster
import warnings
from langchain.vectorstores import Pinecone
import pinecone
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import time
load_dotenv()

ASTRA_DB_SECURE_BUNDLE_PATH ="G:/GENAI/Medical_chat_bot/src/secure-connect-medical-bot.zip"
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=os.getenv('LANGCHAIN_API_KEY')
os.environ["LANGCHAIN_PROJECT"]="Medical_chatbot"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"
# ASTRA_DB_APPLICATION_TOKEN=os.getenv("ASTRA_DB_APPLICATION_TOKEN")
# ASTRA_DB_ID=os.getenv("ASTRA_DB_ID")
# ASTRA_DB_KEYSPACE=os.getenv("ASTRA_DB_KEYSPACE")
# ASTRA_DB_API_ENDPOINT=os.getenv("ASTRA_DB_API_ENDPOINT")
# ASTRA_DB_CLIENT_ID=os.getenv("ASTRA_DB_CLIENT_ID")
# ASTRA_DB_CLIENT_SECRET=os.getenv("ASTRA_DB_CLIENT_SECRET")

ASTRA_DB_APPLICATION_TOKEN="AstraCS:aqNqadQtzDPbCQJBENaWANnt:09203ed247000c81ac32d8a822f3f14e0d8cffa38c5ae3c3a94868f487a9f016"
ASTRA_DB_ID="94dd9f3c-c237-4e95-b91f-6b214d4ae99f-1"
ASTRA_DB_KEYSPACE="medical_bot"
ASTRA_DB_API_ENDPOINT="https://94dd9f3c-c237-4e95-b91f-6b214d4ae99f-us-east1.apps.astra.datastax.com"
ASTRA_DB_CLIENT_ID="aqNqadQtzDPbCQJBENaWANnt"
ASTRA_DB_CLIENT_SECRET="kEU5WxHs+.fUzJ0Ci8FYIBl1.yK7b+bXct1+UTvpcYUG77KrKPKpzJrGfKXT0BAuQ4r1_,W+Ko0Zder00_hetHMszrE6y4sAAlB7YJ7fc+,zpvU.n-uCTGKrUoCqe95H"
ASTRA_DB_TABLE='medical_chat_bot_demo'


# cassio.init(token=ASTRA_DB_APPLICATION_TOKEN,database_id=ASTRA_DB_ID,secure_connect_bundle=ASTRA_DB_SECURE_BUNDLE_PATH)

# cloud_config = {
#     'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH
# }
# print('im from helper.py')
def doc_loader(pdf_reader):
    # print('im from doc_loc fn')
    encode_kwargs = {'normalize_embeddings': True}
    huggigface_embeddings=HuggingFaceBgeEmbeddings(
    model_name='BAAI/bge-small-en-v1.5',
    # model_name='sentence-transformers/all-MiniLM-16-v2',
    model_kwargs={'device':'cpu'},
    encode_kwargs=encode_kwargs)


    loader=PyPDFLoader(pdf_reader)
    documents=loader.load_and_split()
    # print('iam after documents loader called')
   
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
    final_documents=text_splitter.split_documents(documents)
    # print('iam after final_documents  called',final_documents)
   
    os.environ['PINECONE_API_KEY'] =  os.environ['pinecone']
    os.environ['PINECONE_API_ENV'] =  "pdf_query_db"
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    
    index = pc.Index("pdf-query-index")
    namespace = "pdf_query_medical"
    
    def namespace_exists(index, namespace):
        try:
            stats = index.describe_index_stats()
            return namespace in stats['namespaces']
        except pinecone.core.client.exceptions.NotFoundException:
            return False
    if namespace_exists(index, namespace):
        print(f"Namespace '{namespace}' exist.")
        pinecone_vector_store = PineconeVectorStore(embedding=huggigface_embeddings,index_name="pdf-query-index", namespace=namespace)
        # pinecone_vector_store = index.query(f"SELECT * FROM {namespace}")
        # return pinecone_vector_store
    else:
        print(f"Namespace '{namespace}' does not exist. It will be created upon upsertion.")


        pinecone_vector_store=PineconeVectorStore(embedding=huggigface_embeddings,index_name="pdf-query-index",namespace=namespace)

        pinecone_vector_store.add_documents(final_documents)

    return pinecone_vector_store
   
    
    

# def doc_loader(pdf_reader):
#     # print('im from doc_loc fn')
#     encode_kwargs = {'normalize_embeddings': True}
#     huggigface_embeddings=HuggingFaceBgeEmbeddings(
#     model_name='BAAI/bge-small-en-v1.5',
#     # model_name='sentence-transformers/all-MiniLM-16-v2',
#     model_kwargs={'device':'cpu'},
#     encode_kwargs=encode_kwargs)


#     loader=PyPDFLoader(pdf_reader)
#     documents=loader.load_and_split()
#     # print('iam after documents loader called')
   
#     text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
#     final_documents=text_splitter.split_documents(documents)
#     # print('iam after final_documents  called',final_documents)
   
#     astrasession = Cluster(
#     cloud={"secure_connect_bundle": ASTRA_DB_SECURE_BUNDLE_PATH},
#     auth_provider=PlainTextAuthProvider("token", ASTRA_DB_APPLICATION_TOKEN),
#     ).connect()
   
   
    
#     check_table_query = f"""
#     SELECT table_name FROM system_schema.tables 
#     WHERE keyspace_name='{ASTRA_DB_KEYSPACE}' AND table_name='{ASTRA_DB_TABLE}';
#     """

  
#     try:

#         result = astrasession.execute(check_table_query)
       
#         if result.one():
#             return_query=f""" select * from '{ASTRA_DB_KEYSPACE}'.'{ASTRA_DB_TABLE}'; """
#             astra_vector_store=astrasession.execute(return_query)
#             return astra_vector_store
       
       
#         else:
            
#             print(f"Table {ASTRA_DB_KEYSPACE}.{ASTRA_DB_TABLE} does not exist. Try to create table.")
    
    
#             astra_vector_store=Cassandra(
#             embedding=huggigface_embeddings,
#             table_name='medical_bot_demo',
#             session=astrasession,
#             keyspace=ASTRA_DB_KEYSPACE
#             )

        
#             astra_vector_store.add_documents(final_documents)
#             if astra_vector_store:
#                 print("Vector store created successfully")

#             return astra_vector_store
#     except Exception as e:
#         print(f"Error checking/creating keyspace: {e}")

