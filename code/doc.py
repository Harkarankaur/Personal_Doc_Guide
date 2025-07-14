from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser 
from langchain_community.llms import Ollama
import streamlit as st 
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import shutil

load_dotenv()


##tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

llm = Ollama(model="llama2")
embeddings = OllamaEmbeddings(model="llama2")
##creating bot
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","you are a helpful assistant. Please provide reaponse to the user queries"),
        ("user","Question:{question}")
    ]
)
#streamlit
st.set_page_config(page_title="Personal Guide", layout="wide")
st.title("Guide")
input_text=st.text_input("search the topic you want")
uploaded_file = st.file_uploader("Upload a PDF document", type=["pdf"], key="pdf_upload")
#open ai llm call

output_parser=StrOutputParser()
##chain
chain=prompt|llm|output_parser

            ##Document
if uploaded_file is not None:
            with st.spinner("Reading and indexing your document..."):
    
        # Save file temporarily
                file_path = os.path.join("temp", uploaded_file.name)
                os.makedirs("temp", exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.read())
        # Load and split
                loader = PyPDFLoader(file_path)
                docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)

        # Embed and store
                vectordb = FAISS.from_documents(chunks, embeddings)

        # QA Chain
                qa = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=vectordb.as_retriever()
            )

                st.success("Document processed. Ask your questions..")

        # User input
            if input_text is not None:
                with st.spinner("Searching your document"):
                    result = qa.run(input_text)
                    st.markdown(f"**Answer:** {result}")
            try:
                    if os.path.exists(file_path):
                        os.remove(file_path)  # delete the uploaded file

        # If you want to delete the whole folder:
                    if os.path.exists("temp") and not os.listdir("temp"):
                        os.rmdir("temp")  # delete the folder if it's empty


            except Exception as e:
                st.warning(f" Cleanup failed: {e}")
elif uploaded_file is None:
            with st.spinner("Searching..."):
                st.write(chain.invoke({'question':input_text}))
else: 
            st.info(" Please upload a PDF file in the side menu or ask any text query to get started.")
