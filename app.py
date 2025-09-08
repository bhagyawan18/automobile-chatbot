import os
import streamlit as st

from PyPDF2 import PdfReader
import google.generativeai as genai

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import asyncio
from dotenv import load_dotenv

load_dotenv()
gemini_api_key = "AIzaSyBuwEl50FypC2mQly5oCfecHP7qd8sMbdM"

@st.cache_data(show_spinner=False)
def get_pdf_text(pdf_docs):
    """
    Extracts text from a list of PDF documents.

    Args:
        pdf_docs (List[IO]): A list of PDF documents.

    Returns:
        str: The extracted text from all the PDF documents.

    This function uses the PyPDF2 library to extract text from each page of the PDF documents. It iterates over each PDF document in the given list and reads the document using the PdfReader class. Then, it iterates over each page in the PDF document and extracts the text from each page using the extract_text method. The extracted text from all the pages is concatenated and returned as a single string.

    Note: This function uses the st.cache_data decorator from the Streamlit library to cache the result of the function, which improves performance by avoiding redundant computations.

    Example:
        >>> pdf_docs = [IO('document1.pdf'), IO('document2.pdf')]
        >>> get_pdf_text(pdf_docs)
        'Extracted text from document1.pdf\nExtracted text from document2.pdf'
    """
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

@st.cache_data(show_spinner=False)
def get_text_chunks(text):
    """
    Split the given text into chunks using the RecursiveCharacterTextSplitter.

    Args:
        text (str): The text to be split into chunks.

    Returns:
        List[str]: A list of text chunks.

    This function uses the RecursiveCharacterTextSplitter to split the given text into chunks. The chunk size and overlap are set to 5000 and 500 respectively. The function returns a list of text chunks.

    Example:
        >>> text = "This is a sample text."
        >>> get_text_chunks(text)
        ['This is a sample text.']
    """
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
    chunks = text_splitter.split_text(text)
    return chunks

@st.cache_data(show_spinner=False)
def get_vector_store(text_chunks):
    """
    Generate a vector store from the given text chunks using GoogleGenerativeAIEmbeddings and FAISS.

    Args:
        text_chunks (List[str]): A list of text chunks.

    Returns:
        None

    This function uses the GoogleGenerativeAIEmbeddings model to generate embeddings for each text chunk. It then creates a vector store using FAISS from the embeddings and saves it locally to "faiss_index".

    Note:
        - This function uses the st.cache_data decorator from the Streamlit library to cache the result of the function, which improves performance by avoiding redundant computations.
        - The embeddings model is set to "models/embedding-001".

    Example:
        >>> text_chunks = ["Chunk 1", "Chunk 2", "Chunk 3"]
        >>> get_vector_store(text_chunks)
    """
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=gemini_api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """
    Returns a conversational chain for answering questions about the electric vehicle (EV) industry.
    
    This function creates a conversational chain using the ChatGoogleGenerativeAI model, which is a language model specialized in the EV industry. The chain is initialized with a prompt template that defines the structure of the conversation. The prompt template includes placeholders for the context and question, which are replaced with the actual values when the chain is used to generate an answer.
    
    The model is configured with a temperature of 0.3, which controls the randomness of the generated responses. The chain is loaded using the load_qa_chain function, which creates a question-answering chain based on the provided model and prompt.
    
    The function returns the created conversational chain.
    
    Parameters:
    None
    
    Returns:
    chain (ConversationalChain): The conversational chain for answering questions about the EV industry.
    """
    
    prompt_template = """
    Answer the question as if you are an Automobile engineer specializing in the EV (Electric Vehicle) industry,
    ensuring you incorporate detailed technical knowledge about EV car technology and please provide a better understandable answer to the user. 
    If the answer is not available in the provided context,
    simply say, "The answer is not available in the context." Do not provide incorrect information.\n\n
    Context:\n {context}\n
    Question: \n{question}\n

    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, api_key=gemini_api_key)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

async def user_input(user_question):
    """
    Asynchronously processes a user question and returns the response from a conversational chain.

    Args:
        user_question (str): The question entered by the user.

    Returns:
        str: The response generated by the conversational chain.

    This function first creates an instance of GoogleGenerativeAIEmbeddings with the specified model.
    Then, it loads a FAISS index from a local file named "faiss_index" using the embeddings.
    The function then performs a similarity search on the index using the user_question.
    The resulting documents are stored in the docs variable.

    Next, the function creates a conversational chain using the get_conversational_chain function.
    The chain is then used to generate a response to the user_question using the docs and user_question as input.
    The generated response is returned as a string.

    Note:
        - This function is asynchronous and should be called using an async function or awaited.
        - The FAISS index is loaded using the allow_dangerous_deserialization parameter.
        - The conversational chain is generated using the get_conversational_chain function.
    """
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)

    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)

    return response["output_text"]

def main():
    """
    The main function that runs the EV Expert application.

    This function sets the page configuration to "EV Expert" and adds a header to the page. It then displays a text input field where the user can ask a question from the PDF files. If the user enters a question, the function processes the question by calling the `user_input` function asynchronously. The response from the `user_input` function is displayed on the page.

    The function also includes a sidebar with a title "Menu:" and a file uploader to upload PDF files. If the user clicks the "Submit & Process" button and there are PDF files uploaded, the function processes the PDF files by calling the `get_pdf_text`, `get_text_chunks`, and `get_vector_store` functions. If the PDF files are successfully processed, a success message is displayed. Otherwise, a warning message is displayed.

    This function does not take any parameters and does not return any values.

    Example usage:
        main()
    """
    
    st.set_page_config("EV Expert")
    st.header("EV Expert")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        with st.spinner("Processing your question..."):
            response = asyncio.run(user_input(user_question))
            st.write("Reply: ", response)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            if pdf_docs:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    get_vector_store(text_chunks)
                    st.success("Done")
            else:
                st.warning("Please upload PDF files first.")

if __name__ == "__main__":
    main()
