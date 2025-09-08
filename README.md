# Gemini Powered Car Manual and Automobiles Chatbot

Gemini Powered Car Manual and Automobiles Chatbot is a Streamlit application that leverages Google Generative AI Gemini and LangChain for conversational question-answering based on Car manual PDF. This chatbot allows users to ask questions related to the content of uploaded PDF files.

## Features
- **Conversational Chatbot:** Utilizes Google Generative AI for conversational question-answering.
- **PDF Processing:** Extracts text from uploaded PDF files.
- **Vectorization:** Converts text chunks into embeddings for efficient similarity search.
- **User Interaction:** Users can ask questions related to the content of PDF files.

## Usage
1. **Installation**
   - Clone the repository:

     ```bash
     git clone https://github.com/AJAmit17/automobile-chatbot.git
     cd automobile-chatbot
     ```

2. **Create and Activate a Virtual Environment**
   - Create a virtual environment:

     ```bash
     python -m venv venv
     ```

   - Activate the virtual environment:
     - On Windows:

       ```bash
       venv\Scripts\activate
       ```
     - On macOS/Linux:

       ```bash
       source venv/bin/activate
       ```

3. **Install Dependencies**
   - Install the required dependencies:

     ```bash
     pip install -r requirements.txt
     ```

4. **Running the Application**
   - Run the Streamlit application:

     ```bash
     streamlit run app.py
     ```

## Usage Instructions

1. Upload PDF files using the file uploader.
2. Ask questions related to the content of the PDF files in the text input box.
3. Click "Submit & Process" to extract text from the uploaded PDF files and process the user's question.
4. The chatbot will provide an answer based on the context of the uploaded PDF files.

## Dependencies
- Streamlit
- PyPDF2
- langchain
- Google Generative AI
- python-dotenv
