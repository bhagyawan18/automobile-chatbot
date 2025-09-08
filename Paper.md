# Research Paper

## Abstract

The Gemini MultiPDF Chatbot is an innovative tool in natural language processing (NLP) that combines Retrieval-Augmented Generation (RAG) techniques with the Gemini Large Language Model. Designed for the automobile industry, this chatbot allows users to interact effortlessly with vehicle user manuals and related PDF documents. By employing RAG methods, the chatbot enhances its ability to retrieve, comprehend, and generate responses based on extensive information from various automobile-related PDFs. This integration of Gemini's language comprehension with RAG ensures a seamless user experience, providing detailed and contextually relevant answers to questions about vehicle features, maintenance, and troubleshooting. This paper discusses the design, implementation, and evaluation of the Gemini MultiPDF Chatbot, highlighting its effectiveness in managing complex automotive information and delivering high-quality conversational interactions focused on vehicle user manuals.

**Keywords:** Retrieval-Augmented Generation, FAISS Index, LangChain, Large Language Models (LLMs), Automobile User Manuals, Natural Language Processing.

---

## I. INTRODUCTION

In recent years, we've witnessed incredible advancements in how computers understand human language, thanks to innovations like Retrieval-Augmented Generation (RAG). RAG combines two powerful techniques: one for retrieving information and another for generating responses. Now, imagine mixing RAG with an ultra-smart language model like Gemini. That’s where the Gemini MultiPDF Chatbot comes into play.

Think of this chatbot as your savvy car manual assistant. It’s like having a knowledgeable friend who’s great at sifting through and explaining details from multiple car manuals and other PDF documents. This introduction will dive into how the chatbot works, what makes it special, and how it can make handling complex car information a breeze.

Gemini models are pros at various NLP tasks, such as summarizing texts, analyzing sentiment, and translating languages. Thanks to their dual-encoder architecture, Gemini models excel across a range of NLP benchmarks, making them the go-to choice for cutting-edge solutions in natural language processing. With the Gemini MultiPDF Chatbot, you get all these capabilities focused on helping you navigate your car’s manual with ease.

## II. METHODOLOGY

The Gemini MultiPDF Chatbot employs advanced NLP techniques to deliver accurate and contextually relevant responses to user queries from multiple PDF documents. This section outlines the methodology used to develop the chatbot, focusing on integrating Retrieval-Augmented Generation (RAG) with the Gemini Large Language Model and utilizing various libraries and tools to achieve this goal.

### A. Data Ingestion and Preprocessing

1. **PDF Text Extraction**:
   - The `get_pdf_text` function extracts text from uploaded PDF documents using the PyPDF2 library. It processes each page of the PDFs and concatenates the extracted text.
   - The function is decorated with `@st.cache_data` to cache the extracted text

2. **Text Chunking**:
   - The extracted text is divided into smaller, manageable segments using the `get_text_chunks` function. This is done with the RecursiveCharacterTextSplitter from LangChain, setting a chunk size of 5000 characters with a 500-character overlap to maintain context between chunks.

### B. Vector Store Creation

3. **Embedding Generation**:
   - The text chunks are transformed into embeddings using the GoogleGenerativeAIEmbeddings model "embedding-001". These embeddings are numerical representations that capture the semantic meaning of the text, enabling efficient similarity searches.

4. **FAISS Index Construction**:
   - To facilitate fast and accurate retrieval of relevant text chunks based on user queries, the FAISS library is used to create a vector store from the generated embeddings.
   - The `get_vector_store` function saves the vector store locally, ensuring it can be quickly loaded and reused.

### C. Conversational Chain Setup

5. **Prompt Template**:
   - A custom prompt template is established to guide the chatbot in responding to user queries. This template ensures that the chatbot provides accurate technical information regarding Electric Vehicles (EVs) and avoids giving incorrect answers when the context lacks sufficient information.

6. **Loading the QA Chain**:
   - The `get_conversational_chain` function initializes a question-answering chain using the ChatGoogleGenerativeAI model "gemini-pro". The chain type "stuff" processes both the context and the user query to produce a response based on the specified prompt template.

### D. User Interaction and Query Processing

7. **User Input Handling**:
    - The `user_input` function manages user queries by loading the FAISS index and conducting a similarity search to find relevant text segments related to the user's question.
    - The identified documents and the user query are then submitted to the conversational chain to generate a response.

8. **Streamlit Interface**:
   - The primary function establishes an intuitive interface using Streamlit. Users can upload multiple PDF files, ask questions, and receive answers.
   - The interface features a sidebar for uploading files and a button to initiate the processing of the PDFs. Once processed, the extracted text is divided into chunks, embedded, and indexed for quick retrieval.

### E. Implementation Workflow

9. **Execution Flow**:
   - The main workflow begins with the user uploading PDF documents. The text extraction and chunking processes are triggered upon submission.
   - When a user asks a question, the chatbot navigates it through the conversational chain, utilizing the precomputed embeddings and FAISS index to deliver an accurate and contextually appropriate response.

![Image](https://media.licdn.com/dms/image/D4D12AQEpIbqAiRYMTQ/article-cover_image-shrink_720_1280/0/1699197471598?e=2147483647&v=beta&t=yWyUPGKqmLdOlhp9K5XxZO7A1vIz4gJs18cNY-B1fyQ)
   
## IV. Formulas

### A. Information Retrieval: Similarity Search

In our chatbot, we use FAISS (Facebook AI Similarity Search) to perform similarity searches. The similarity between a user query and document chunks is measured using cosine similarity.

**Cosine Similarity Formula:**

Given two vectors \(A\) and \(B\), the cosine similarity is defined as:

$$
\text{Cosine Similarity} = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|}
$$

Where:

- \(A \cdot B\) is the dot product of vectors \(A\) and \(B\).
- \(\|A\|\) and \(\|B\|\) are the magnitudes of vectors \(A\) and \(B\).

### B. Text Embeddings: Generating Embeddings

Text embeddings are generated using pre-trained models like Google's `embedding-001`. This process converts text into high-dimensional vectors that capture semantic meaning.

**Vector Embedding Formula:**

If \(\mathbf{X}\) is the input text, the embedding model \(\mathbf{f}\) maps it to a vector space:

$$
\mathbf{E}=f(\mathbf{X})
$$

Where \(\mathbf{E}\) is the embedding vector of the input text \(\mathbf{X}\).

### C. Generative Model: Retrieval-Augmented Generation (RAG)

RAG combines retrieval and generation, typically using a sequence-to-sequence (Seq2Seq) model with an encoder-decoder architecture. The RAG process involves retrieving relevant documents and generating a response based on the retrieved context.

**Seq2Seq Model Formulas:**

1. **Encoder:**
   The encoder processes the input sequence \(\mathbf{X}\) and generates a context vector \(\mathbf{C}\):

   $$
   \mathbf{C}=\text{Encoder}(\mathbf{X})
   $$

2. **Decoder:**
   The decoder generates the output sequence \(\mathbf{Y}\) based on the context vector \(\mathbf{C}\) and previously generated tokens \(\mathbf{Y}_{<t}\):

   $$
   P(\mathbf{Y}_t|\mathbf{Y}_{<t},\mathbf{C})=\text{Decoder}(\mathbf{Y}_{<t},\mathbf{C})
   $$

**Attention Mechanism:**

An attention mechanism allows the model to focus on different parts of the input sequence when generating each token in the output sequence.

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

Where:

- \(Q\) (Query), \(K\) (Key), and \(V\) (Value) are matrices derived from the input sequence.
- \(d_k\) is the dimensionality of the key vectors.

### D. Evaluation Metrics

To evaluate the chatbot’s performance, we use common NLP evaluation metrics, including precision, recall, F1-score, and BLEU score.

**Precision, Recall, and F1-Score:**

1. **Precision:**

   $$
   \text{Precision}=\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
   $$

2. **Recall:**

   $$
    \text{Recall}=\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
   $$

3. **F1-Score:**
   $$
    \text{F1-Score}=2\times\frac{\text{Precision}\times\text{Recall}}{\text{Precision}+\text{Recall}}
   $$

**BLEU Score:**

The BLEU (Bilingual Evaluation Understudy) score evaluates the quality of machine-generated text by comparing it to reference texts.

$$
\text{BLEU}=\text{BP}\cdot\exp\left(\sum_{n=1}^Nw_n\log{p_n}\right)
$$

Where:

- \(\text{BP}\) is the brevity penalty.
- \(p_n\) is the precision of n-grams.
- \(w_n\) is the weight assigned to n-gram precision (typically \(\frac{1}{N}\) for uniform weights).

### E. Application in Chatbot

1. **Similarity Search for Relevant Context:**

   - Compute cosine similarity between user query embeddings and document chunk embeddings.
   - Retrieve the top-k similar chunks based on cosine similarity scores.

2. **Embedding Text Chunks:**

   - Use the embedding model to convert text chunks into high-dimensional vectors.

3. **Generating Responses:**

   - Use the retrieved context and the user query to generate responses using a Seq2Seq model with attention mechanisms.

4. **Evaluation:**
   - Measure the chatbot’s performance using precision, recall, F1-score, and BLEU score against a set of predefined correct responses.

---

### IV. Results

We evaluated our two approaches by examining how effectively human users could find information in automobile user manuals using our chatbot. The evaluation considered various components, including accuracy, relevance of responses, and user satisfaction. The following sections detail the results of implementing and testing our chatbot.

#### A. Accuracy of Responses
To assess accuracy, we tested the chatbot's responses against correct answers extracted from automobile user manuals. The chatbot achieved a high success rate of 92% across all queries. This high precision is largely due to the integration of a Background Retrieval model with the Gemini Large Language Model in a RAG (Retrieval-Augmented Generation) setup.

#### B. Relevance of Responses
We rated the contextual alignment of responses to the information in the relevant user manual pages. The chatbot provided contextually appropriate answers for 95% of the queries, sticking closely to the extracted PDF text chunks.

#### C. User Satisfaction
We conducted a user experience survey with 50 participants who interacted with the chatbot. The survey included open-text questions about the clarity, helpfulness, and overall satisfaction with the chatbot's responses. The results showed that 90% of participants described their experience as either "satisfactory" or "highly satisfactory."

#### D. Processing Efficiency
We measured processing efficiency by evaluating the time it took to process and index PDF documents and generate responses. The chatbot processed and indexed a 100-page PDF document in under 2 minutes on average. It generated responses to user queries in less than 3 seconds, demonstrating both speed and efficiency.

#### E. Case Study: EV User Manual
To test the chatbot in a real-world scenario, we used an Electric Vehicle (EV) user manual. The chatbot effectively handled queries about battery service, a crucial aspect of troubleshooting and charging EVs. The responses were not only accurate but also provided detailed information relevant to the questions.

#### F. Limitations and Challenges
Despite the overall success, the chatbot has some limitations:
- **Context Limitations:** For messages requiring information from multiple nonconsecutive chunks, the chatbot struggled to provide a coherent answer.
- **Out-of-Scope Questions:** When users asked questions not covered in the provided PDF, the chatbot responded with "The answer doesn't exist in this context," which caused some user frustration.
- **Technical Jargon:** The chatbot sometimes provided overly technical explanations, which some users found difficult to understand.

#### G. Future Improvements
To address these limitations, future improvements will focus on:
- **Enhanced Context Aggregation:** Improving the chatbot's ability to aggregate contextual information from non-contiguous chunks to create more coherent answers.
- **Broader Knowledge Integration:** Adding more sources to cover a wider range of queries.
- **Simplified Explanations:** Providing plain English responses for users who might not understand technical details.


### V. References
## References

1. Y. Chang et al., “A Survey on Evaluation of Large Language Models,” Jul. 2023. [Online] Available: [http://arxiv.org/abs/2307.03109](http://arxiv.org/abs/2307.03109)
2. S. Siriwardhana, R. Weerasekera, E. Wen, T. Kaluarachchi, R. † Rajib, and S. Nanayakkara, “Improving the Domain Adaptation of Retrieval Augmented Generation (RAG) Models for Open Domain Question Answering,” doi: 10.1162/tacl.
3. W. Yu, “Retrieval-augmented Generation across Heterogeneous Knowledge.”
4. K. Rangan and Y. Yin, “A Fine-tuning Enhanced RAG System with Quantized Influence Measure as AI Judge,” Feb. 2024. [Online] Available: [http://arxiv.org/abs/2402.17081](http://arxiv.org/abs/2402.17081)
5. J. Liuska, “Bachelor’s Thesis- ENHANCING LARGE LANGUAGE MODELS FOR DATA ANALYTICS THROUGH DOMAIN SPECIFIC CONTEXT CREATION,” 2024.
6. G. Gemini Team, “Gemini: A Family of Highly Capable Multimodal Models, 2024.”
