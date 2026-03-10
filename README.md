# Explain Yourself RAG

Explain Yourself RAG is an explainable Retrieval-Augmented Generation (RAG) inspection tool designed to make semantic retrieval systems transparent.
Instead of acting like a black-box chatbot, this system allows users to inspect how document chunks are retrieved, why they are selected, and how similarity scores influence the retrieval process.
The application ingests PDF documents, splits them into chunks, generates vector embeddings, and retrieves the most relevant sections using semantic similarity.

## Features

- Multi-document PDF ingestion
- Semantic retrieval using embeddings
- Transparent similarity score ranking
- Sentence-level highlighting of query matches
- Query-to-chunk similarity visualization
- Semantic chunk clustering using PCA
- Retrieval analytics dashboard
- Exportable retrieved evidence

## How It Works

The system follows a simple Retrieval-Augmented pipeline:
Document Upload → Text Chunking → Embedding Generation → Semantic Retrieval → Evidence Inspection

1. Documents are uploaded as PDFs.
2. Text is split into overlapping chunks.
3. Each chunk is converted into an embedding vector.
4. A user query is embedded into the same vector space.
5. Cosine similarity is used to retrieve the most relevant chunks.
6. The system visualizes retrieval results and analytics.

## Technologies Used

- Streamlit
- SentenceTransformers
- NumPy
- Pandas
- PyPDF
- Scikit-learn

## Installation

Clone the repository and install dependencies.

```bash
pip install -r requirements.txt
