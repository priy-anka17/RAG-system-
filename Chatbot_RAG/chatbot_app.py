import streamlit as st 
import os
import base64
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM 
from transformers import pipeline
import torch 
import textwrap 
from langchain.document_loaders import PyPDFLoader, DirectoryLoader, PDFMinerLoader 
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma 
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA 
#from constants import CHROMA_SETTINGS
from streamlit_chat import message
st.set_page_config(layout="wide")

@st.cache_data
#function to display the PDF of a given file 
def displayPDF(file):
    # Opening file from file path
    with open(file, "rb") as f:
        base64_pdf = base64.b64encode(f.read()).decode('utf-8')

    # Embedding PDF in HTML
    pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="600" type="application/pdf"></iframe>'

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)

@st.cache_resource
def data_ingestion():
    for root, dirs, files in os.walk("docs"):
        for file in files:
            if file.endswith(".pdf"):
                print(file)
                loader = PDFMinerLoader(os.path.join(root, file))
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=500)
    texts = text_splitter.split_documents(documents)
    #create embeddings here
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    #create vector store here
    db = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory, client_settings=CHROMA_SETTINGS)
    db.persist()
    db=None 

@st.cache_resource
def llm_pipeline():
    pipe = pipeline(
        'text2text-generation',
        model = base_model,
        tokenizer = tokenizer,
        max_length = 256,
        do_sample = True,
        temperature = 0.3,
        top_p= 0.95,
    )
    local_llm = HuggingFacePipeline(pipeline=pipe)
    return local_llm

@st.cache_resource
def qa_llm():
    llm = llm_pipeline()
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="db", embedding_function = embeddings, client_settings=CHROMA_SETTINGS)
    retriever = db.as_retriever()
    qa = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = retriever,
        return_source_documents=True
    )
    return qa

def process_answer(instruction):
    response = ''
    instruction = instruction
    qa = qa_llm()
    generated_text = qa(instruction)
    answer = generated_text['result']
    return answer


# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))


def main():
    st.markdown("<h1 style='text-align:center; color: blue;'> Chat with your PDF </h1>",
    unsafe_allow_html=True)

    st.markdown("<h3 style='text-align:center; color: grey;'> Built by Priyanka :) </h3>", unsafe_allow_html=True)

    st.markdown("<h2 stye='text-align:center; color:red;'> Upload your PDF Below</h2>", unsafe_allow_html=True)

    uploaded_file=st.file_uploader("", type=["pdf"])

    if uploaded_file is not None:
        file_details={
            "name": uploaded_file.name,
            "type": uploaded_file.type,
            "size": uploaded_file.size
        }
        filepath="docs/"+uploaded_file.name
        with open(filepath, "wb") as temp_file:

            temp_file.write(uploaded_file.read())

        col1, col2=st.columns([1, 2])
        with col1:
            st.markdown("<h2 style='text-align:center; color: red;'> PDF Details </h2>",
            unsafe_allow_html=True)
            st.write(file_details)
            st.markdown("<h2 style='text-align:center; color: red;'> PDF Preview </h2>",
            unsafe_allow_html=True)
            displayPDF(filepath)

        with col2:
            with st.spinner("Embeddings are in process....."):
                ingested_data=data_ingestion()
            st.success('Embeddings are created successfully!')
            st.markdown("<h2 style='text-align: center; color: grey; '> Chat Here </h2>",
            unsafe_allow_html=True)


if __name__=="__main__":
    main()