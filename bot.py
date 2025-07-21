# ANSWER QUERY RELATED TO TXT,DOCS,PDF FILES SINGLY




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
from docx import Document
from langchain.document_loaders import Docx2txtLoader
from langchain.document_loaders import TextLoader
import hashlib
from langchain_core.runnables import RunnableLambda


load_dotenv()



##tracking
os.environ["LANGCHAIN_API_KEY"]=os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"]="true"

llm = Ollama(model="llama2")
embeddings = OllamaEmbeddings(model="llama2")
##creating chat bot
prompt=ChatPromptTemplate.from_messages(
    [
       (
            "system",
            "You are a helpful assistant. If a context is provided, answer the question using only the context. "
            "If the answer cannot be found in the context, say: 'I cannot answer that based on the document.' "
            "If no context is given, answer the question using your general knowledge."
        ),
        ("user", "Context:\n{context}\n\nQuestion: {question}")
    
    ]
)
#streamlit
st.set_page_config(page_title="Personal Guide", layout="wide")
st.title("Guide")
def clear_input():
    st.session_state["query_input"]=""

input_text=st.text_input("search the topic you want")

with st.sidebar:
    st.title("Docs for query:")
    uploaded_file = st.file_uploader("Upload the PDF document", type=["pdf","docx","txt"], key="doc_upload")
# Utility: Generate hash for file
def get_file_hash(file):
    file.seek(0)
    content = file.read()
    file.seek(0)
    return hashlib.md5(content).hexdigest()

# Utility: Save uploaded file temporarily
def save_uploaded_file(uploaded_file, save_path):
    with open(save_path, "wb") as f:
        f.write(uploaded_file.read())

#open ai llm call
output_parser=StrOutputParser()
##chain
chain=prompt|llm|output_parser
#text query
if uploaded_file is None: 
    if input_text is None:
        st.info(" Please upload a PDF file & ask query to go.")
if uploaded_file is None:
    if input_text:
        with st.spinner("searching"):
            st.write(chain.invoke({'question':input_text}))
##Document
if uploaded_file and input_text:
        with st.spinner("Processing your document query..."):
            file_hash = get_file_hash(uploaded_file)
            vectordb_path = f"vector_cache/{file_hash}"
            os.makedirs("temp", exist_ok=True)
            os.makedirs("vector_cache", exist_ok=True)
            file_name = uploaded_file.name
            # Save file temporarily
            file_path = os.path.join("temp", uploaded_file.name)
            save_uploaded_file(uploaded_file, file_path)
            #end the query
            def end_query():
                try:
                    if os.path.exists(file_path):
                        os.remove(file_path)  # delete the uploaded file
                # If you want to delete the whole folder:
                        if os.path.exists("temp") and not os.listdir("temp"):
                            os.rmdir("temp")  # delete the folder if it's empty
                    if os.path.exists("vector_cache") :
                        shutil.rmtree("vector_cache")
                except Exception as e:
                    st.warning(f" Cleanup failed: {e}")
            def clear():
                clear_input()
                end_query()
            st.info("Please re-upload to restart")    
            st.button("End query for this document",on_click=clear)

            # Load and split
            if os.path.exists(vectordb_path):
                vectordb = FAISS.load_local(vectordb_path, embeddings,allow_dangerous_deserialization=True )
            else:
                if file_name.endswith(".pdf"):
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                elif file_name.endswith(".docx"):      
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                elif file_name.endswith(".txt"):      
                    loader =TextLoader(file_path)
                    docs = loader.load()

                splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                chunks = splitter.split_documents(docs)
            
            # Embed and store
                vectordb = FAISS.from_documents(chunks, embeddings)
                vectordb.save_local(vectordb_path)

            st.success("Document processed. Answering your questions ")
            # QA Chain
            retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            retrieved_docs = retriever.get_relevant_documents(input_text)
            
            # simple length check or presence of docs
            if not retrieved_docs:
                st.warning(" Your query is not related to the uploaded document.")
            else:
                context = "\n\n".join(doc.page_content for doc in retrieved_docs)

              # Run the combined chain
                response = chain.invoke({
                "context": context,
                "question": input_text
                })
                st.markdown(f"**Answer:** {response}")

           


