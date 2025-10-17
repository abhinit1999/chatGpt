
# importing require libraries
import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# loading env file and API key
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# defining model
llm = ChatGroq(
    model_name="openai/gpt-oss-20b",
    groq_api_key=groq_api_key
)

# Designing user Interface with Stremalit
st.title("RAG application ChatGPT")

# upload documents
uploaded_file = st.file_uploader("upload PDF", type=['PDF'])

# ask_button = False
# handling user's uploaded file/documents
if uploaded_file:

    # adding sppiner
    with st.spinner("Uploading file in progress...."):


        #saving file locally ( in the same directory)
        os.makedirs("Temp_file",exist_ok=True)
        temp_path = os.path.join("Temp_file", uploaded_file.name) #c\chatGPT\temp_file - incase you are saving file in other directory
        with open(temp_path,'wb') as file:
            file.write(uploaded_file.getbuffer())
        
        #handling uploaded file and applying splitter/chunking
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        #chunking
        splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
        splited_chunks = splitter.split_documents(docs)

        # defining HuggingFaceEmbedings
        embeddings = HuggingFaceEmbeddings(model_name="all-miniLM-L6-V2")
        DB = FAISS.from_documents(splited_chunks,embeddings)
        DB.save_local("FAISS_DB_Store")

        # creating retriever
        retriever= DB.as_retriever(search_type="similarity", search_kwargs={'k':5})

        st.success("PDF uploaded... Now start interacting with your Q&A")
        

        


# creating Prompt

prompt = PromptTemplate(

    template=''' 
                You are a helpful tutor and your task is to provide relavent information or transcript based on the 
                uploaded files only. If context is outside of the uploaded file then simply say - "I don't know about this"

                context:{context}
                question:{question}

                ''',

                input_variables=['context','question']
)


# handling user query/ question 
ask_button = False
if uploaded_file:
    st.subheader("Ask any question about the uploaded file/PDF")
    input_question = st.text_input("User Question")

    ask_button = st.button("...ASK...")
# handling aks button

if ask_button:
    with st.spinner("search in....."):
        retrievied_docs = retriever.invoke(input_question)
        final_prompt = prompt.invoke({'context':retrievied_docs,'question':input_question})
        res = llm.invoke(final_prompt)

        #displaying final_answer:
        st.markdown(f"AI Answer:  {res.content}")





