import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
import os
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain

# Load environment variables
load_dotenv()
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
openai_api_key = os.getenv('OPENAI_API_KEY')

# Streamlit app title
st.title("DocBot")

# File uploader for PDF
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Load the PDF document
    loader = PyPDFLoader('temp.pdf')
    loaded_doc = loader.load()

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=600, chunk_overlap=40)
    all_splits = text_splitter.split_documents(loaded_doc)

    # Initialize embeddings and vector store
    embeddings = OpenAIEmbeddings(api_key=openai_api_key)
    db = FAISS.from_documents(loaded_doc, embeddings)

    # Initialize the language model
    llm = Ollama(model="llama3.2")

    # Create the prompt template
    prompt = ChatPromptTemplate.from_template("""Answer the following question strictly only based on the provided document.
                                              <context>
                                              {context}
                                              Question : {input}""")

    # Create the document chain and retriever
    chain = create_stuff_documents_chain(llm, prompt)
    retriever = db.as_retriever()

    # Create the retrieval chain
    retrieval_chain = create_retrieval_chain(retriever, chain)

    # Text input for the question
    question = st.text_input("Enter your question:")

    if st.button("Get Answer"):
        if question:
            # Get the response from the retrieval chain
            response = retrieval_chain.invoke({"input": question})
            answer = response['answer'].split('\n')

            # Display the answer
            st.subheader("Answer:")
            for line in answer:
                st.write(line)
        else:
            st.write("Please enter a question.")
else:
    st.write("Please upload a PDF file.")