from flask import Flask,render_template,jsonify,request
from src.helper import *
from src.prompt import *

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
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from cassandra.auth import PlainTextAuthProvider
import tempfile
import cassio
from PyPDF2 import PdfReader
from cassandra.cluster import Cluster
import warnings
warnings.filterwarnings("ignore")
import os
from dotenv import load_dotenv
import time
load_dotenv()
app = Flask(__name__)

groq_api_key=os.environ['GROQ_API_KEY']
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]="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"
os.environ["LANGCHAIN_PROJECT"]="medical_bot"
os.environ["LANGCHAIN_ENDPOINT"]="https://api.smith.langchain.com"

prompt=PromptTemplate(template=prompt_template, input_variables=["context", "question"])
# print(PROMPT)
llm=ChatGroq(groq_api_key=groq_api_key,model_name="mixtral-8x7b-32768")
file_path="G:/GENAI/Medical_chat_bot/data/Medical_book.pdf"
pinecone_vector_store=doc_loader(file_path)

print(type(pinecone_vector_store))
def generate_response(llm,prompt,pinecone_vector_store,question):

    # print('HELLO!Im from gen reponse fn')
    document_chain=create_stuff_documents_chain(llm,prompt)
    # print('document chain:',prompt)
    retriever=pinecone_vector_store.as_retriever(search_type="similarity",search_kwargs={"k":5})
    # print('HELLO!Im  after retriever')
    retrieval_chain=create_retrieval_chain(retriever,document_chain)
    # print('HELLO!Im  after retrieval chain')
    response=retrieval_chain.invoke({"input":question})
    # print('im response from fn',response)
    return response

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    question = msg
    print(question)
    result=generate_response(llm,prompt,pinecone_vector_store,question)
    # print("Response : ", result['answer'])
    return result['answer']

    
if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 8080, debug= True)
    
