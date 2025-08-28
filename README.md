# 📚 Questionary Chatbot: Advanced RAG Pipeline with LangChain and FAISS

Welcome to my **Questionary Chatbot** project! This project showcases a Retrieval-Augmented Generation (RAG) pipeline built with **LangChain**, leveraging advanced techniques to create a robust question-answering system. Compared to my previous RAG experiment (`Simple_RAG`), this project introduces new methods, including a retrieval chain, prompt engineering, and optimized vector storage. Below, I detail the components used, improvements over the previous project, and the innovative techniques applied to build this chatbot.

---

## 🌟 Project Overview

The **Questionary Chatbot** processes a PDF resume (`Jasweer_Tadikonda_MLE_Engineer.pdf`) to answer queries about my professional experience, skills, and projects. It uses LangChain’s RAG framework, combining document retrieval with generative responses powered by the **Llama2** model (via Ollama). The pipeline ingests documents, chunks them, stores embeddings in a **FAISS** vector store, and employs a retrieval chain with a custom prompt template for precise, context-driven answers.

---

## 🛠️ Project Structure

- **`retriever.ipynb`**: Jupyter notebook containing the RAG pipeline, including data ingestion, chunking, embedding storage, and retrieval chain setup.
- **`Jasweer_Tadikonda_MLE_Engineer.pdf`**: Input resume document for querying professional details.
- **Dependencies**: LangChain, FAISS, Ollama, PyPDFLoader, and related libraries.

---

## 🎯 Techniques and Components Used

### 1. Data Ingestion with PyPDFLoader
- **What I Used**: Employed `PyPDFLoader` to extract text from the PDF resume, preserving metadata like source and page details.
- **Why It’s Effective**: PyPDFLoader handles complex PDF structures, ensuring accurate text extraction for professional documents like resumes.
- **Comparison to Previous**: Unlike `Simple_RAG`, which used multiple loaders (TextLoader, WebBaseLoader), this project focuses solely on PDF ingestion, streamlining the pipeline for a specific use case (resume-based Q&A).

### 2. Text Chunking with RecursiveCharacterTextSplitter
- **What I Used**: Applied `RecursiveCharacterTextSplitter` with a `chunk_size` of 1000 and `chunk_overlap` of 20 to split the resume into manageable segments.
- **Why It’s Effective**: Smaller chunks improve retrieval granularity, while overlap ensures context retention, critical for coherent answers.
- **Comparison to Previous**: Consistent with `Simple_RAG`, but optimized chunk size for resume content, reducing noise from overly large chunks.

### 3. Embeddings and FAISS Vector Store
- **What I Used**: Used `OllamaEmbeddings` with the Llama2 model to generate embeddings, stored in a **FAISS** vector store for efficient similarity searches.
- **Why It’s Effective**: FAISS leverages vector quantization for fast, scalable retrieval, ideal for large datasets or production environments.
- **Comparison to Previous**: Replaced the custom `GemmaEmbeddings` wrapper from `Simple_RAG` with `OllamaEmbeddings`, which is natively supported by LangChain, reducing complexity. FAISS remains the preferred vector store over Chroma for its superior speed and scalability.

### 4. Retrieval Chain with Prompt Engineering
- **What I Used**: Created a retrieval chain using `create_retrieval_chain`, combining a `ChatPromptTemplate` with the Llama2 model and FAISS retriever. The prompt instructs the model to answer step-by-step based solely on provided context.
  ```python
  prompt = ChatPromptTemplate.from_template("""
  Answer the following question based only on the provided context.
  Think step by step before providing a detailed answer.
  I will provide reward points based on user find the answer helpful.
  <context>
      {context}
  <context>
  Question: {input}
  """)
  ```
- **Why It’s Effective**: The custom prompt ensures focused, context-driven responses, reducing hallucinations. The retrieval chain integrates document retrieval with generation, streamlining the Q&A process.
- **Comparison to Previous**: `Simple_RAG` lacked a generative component, relying solely on similarity searches. The retrieval chain adds a generative layer, enabling natural language answers, a significant improvement for user interaction.

### 5. Querying with Similarity Search
- **What I Used**: Implemented `similarity_search_with_score` to retrieve relevant document chunks, tested with queries like “Tell me about Jasweer?” and “Tell me your Fraud Detection Project?”
- **Why It’s Effective**: Similarity scores allow ranking of retrieved documents, ensuring the most relevant chunks are used for generation.
- **Comparison to Previous**: Enhanced from `Simple_RAG` by integrating scored retrieval into a full RAG pipeline, improving answer accuracy and relevance.

---

## 🚀 Improvements Over Previous Project (`Simple_RAG`)

1. **Generative Capabilities**:
   - **Previous**: Focused on retrieval only, returning raw document chunks without natural language processing.
   - **Current**: Integrates a retrieval chain with Llama2 for generative answers, making the chatbot more user-friendly and interactive.

2. **Simplified Embeddings**:
   - **Previous**: Used a custom `GemmaEmbeddings` wrapper for a non-standard model, requiring manual implementation.
   - **Current**: Leveraged `OllamaEmbeddings` for seamless integration with Llama2, reducing development overhead and improving compatibility.

3. **Focused Data Source**:
   - **Previous**: Handled multiple data sources (text, PDF, web), increasing complexity.
   - **Current**: Streamlined to PDF ingestion, optimizing for resume-based Q&A, which aligns with the chatbot’s purpose.

4. **Prompt Engineering**:
   - **Previous**: No generative prompt, limiting output to retrieved text.
   - **Current**: Custom `ChatPromptTemplate` ensures structured, context-driven responses, enhancing answer quality.

5. **Production-Ready Pipeline**:
   - **Previous**: Experimental, focused on testing loaders and vector stores.
   - **Current**: End-to-end RAG pipeline with retrieval and generation, suitable for real-world Q&A applications.

---

## 🧠 New Techniques Introduced

1. **Retrieval-Augmented Generation (RAG) Chain**:
   - Used `create_retrieval_chain` to combine retrieval and generation, enabling the chatbot to fetch relevant documents and generate coherent answers. This is a leap from the retrieval-only approach in `Simple_RAG`.
   - Example: Querying “Tell me about Jasweer?” retrieves resume chunks and generates a detailed summary using Llama2.

2. **Prompt Engineering for Step-by-Step Reasoning**:
   - Designed a prompt that instructs the LLM to “think step by step,” improving response clarity and reducing errors. The inclusion of a reward points mention encourages user feedback, a novel approach for iterative improvement.

3. **OllamaEmbeddings for Local Inference**:
   - Adopted `OllamaEmbeddings` for Llama2, leveraging local inference to reduce dependency on cloud APIs, enhancing privacy and cost-efficiency compared to cloud-based embeddings in previous projects.

4. **Streamlined FAISS Integration**:
   - Optimized FAISS setup for resume data, focusing on fast retrieval with scored results, improving over the dual Chroma/FAISS evaluation in `Simple_RAG`.

---

## 🔧 How to Use the Project

### Prerequisites
- Python 3.8+
- Install dependencies: `pip install langchain langchain-community faiss-cpu pypdf ollama`
- Install and configure Ollama with the Llama2 model (`ollama pull llama2`).
- Place `Jasweer_Tadikonda_MLE_Engineer.pdf` in the project directory.

### Setup Instructions
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Notebook**:
   - Launch Jupyter: `jupyter notebook retriever.ipynb`
   - Execute cells to ingest the PDF, chunk documents, create embeddings, and set up the retrieval chain.

4. **Query the Chatbot**:
   - Run queries like `"Tell me about Jasweer?"` or `"Tell me your Fraud Detection Project?"` using `retrieval_chain.invoke({"input": "your_query"})`.
   - View generated answers based on retrieved resume content.

---

## 🎨 Customizing the Project

1. **Change the LLM**:
   - Replace Llama2 with another Ollama model (e.g., `gemma`) by updating `Ollama(model="gemma")`.
   - Alternatively, integrate a cloud-based LLM like `ChatOpenAI` for enhanced performance.

2. **Enhance the Prompt**:
   - Modify the `ChatPromptTemplate` to include additional instructions (e.g., tone, length) or support multi-turn conversations.

3. **Add More Data Sources**:
   - Extend ingestion to include text or web data using `TextLoader` or `WebBaseLoader`, as in `simplerag.ipynb`.
   - Example: `loader = TextLoader("additional_file.txt")`.

4. **Optimize Retrieval**:
   - Adjust `chunk_size` or `chunk_overlap` in `RecursiveCharacterTextSplitter` for better retrieval accuracy.
   - Experiment with FAISS parameters (e.g., index types) for faster searches.

5. **Deploy as a Web App**:
   - Integrate the pipeline with FastAPI or Streamlit (as in my previous LangServe project) for a user-friendly interface.

---

## 🧠 Why This Approach?

- **Retrieval Chain**: Combines retrieval and generation for a complete Q&A system, surpassing the retrieval-only focus of `Simple_RAG`.
- **OllamaEmbeddings**: Simplifies embedding generation with native LangChain support, avoiding custom wrappers.
- **FAISS**: Chosen for its speed and scalability, making it ideal for production-ready applications.
- **Prompt Engineering**: Ensures precise, context-driven answers, improving user experience.
- **Focused Scope**: Targeting resume-based Q&A reduces complexity while showcasing RAG’s power for professional use cases.

---

## 🎉 Key Takeaways

This project marks a significant evolution from my previous RAG experiments:
- Mastered end-to-end RAG with retrieval chains, enabling natural language Q&A.
- Leveraged prompt engineering to enhance LLM response quality.
- Streamlined embeddings with `OllamaEmbeddings` for efficiency and compatibility.
- Built a scalable, privacy-focused pipeline suitable for real-world applications.

Check out the code on my GitHub: [https://github.com/Jasweer09](https://github.com/Jasweer09). Let’s connect to discuss RAG, AI, or innovative projects! 🚀