import streamlit as st
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
import re
from sklearn.decomposition import PCA


# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(page_title="Explain Yourself RAG", layout="wide")

st.title("Explain Yourself RAG")
st.write("A transparent retrieval inspection system with advanced retrieval analytics.")


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Explain Yourself RAG")

st.sidebar.markdown("""
### System Overview

Explain Yourself RAG is an **Explainable Retrieval Augmented Generation system**.

It provides deep transparency into the retrieval process and allows users to inspect how evidence is selected.

---

### Architecture

1️⃣ Document ingestion  
2️⃣ Text chunking  
3️⃣ Embedding generation  
4️⃣ Semantic retrieval  
5️⃣ Evidence inspection  
6️⃣ Grounded answer generation  

---

### Models Used

Embedding model:
`all-MiniLM-L6-v2`

---

### Core Capabilities

• Multi-document retrieval  
• Evidence-grounded answers  
• Semantic chunk clustering  
• Query-attention visualization  
• Interactive embedding explorer  
• Retrieval transparency dashboard  

---

### Research Applications

• RAG debugging  
• AI explainability  
• Retrieval analysis  
• Semantic search experimentation
""")

top_k = st.sidebar.slider("Chunks Retrieved", 1, 10, 3)


# --------------------------------------------------
# Load Embedding Model
# --------------------------------------------------

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_model()
st.sidebar.success("Embedding model loaded")


# --------------------------------------------------
# Text Chunking
# --------------------------------------------------

def split_text(text, chunk_size=500, overlap=100):

    chunks = []
    start = 0

    while start < len(text):

        end = start + chunk_size
        chunks.append(text[start:end])

        start += chunk_size - overlap

    return chunks


# --------------------------------------------------
# Cosine Similarity
# --------------------------------------------------

def cosine_similarity(a, b):

    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


# --------------------------------------------------
# Sentence Highlighter
# --------------------------------------------------

def highlight_query_sentences(chunk, query):

    sentences = re.split(r'(?<=[.!?]) +', chunk)
    query_words = query.lower().split()

    highlighted = []

    for s in sentences:

        if any(q in s.lower() for q in query_words):
            highlighted.append(f"🔎 **{s}**")
        else:
            highlighted.append(s)

    return " ".join(highlighted)


# --------------------------------------------------
# Upload Multiple PDFs
# --------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:

    all_chunks = []
    chunk_sources = []

    # --------------------------------------------------
    # Extract Text
    # --------------------------------------------------

    for file in uploaded_files:

        reader = PdfReader(file)

        text = ""

        for page in reader.pages:

            extracted = page.extract_text()

            if extracted:
                text += extracted

        chunks = split_text(text)

        all_chunks.extend(chunks)
        chunk_sources.extend([file.name] * len(chunks))

    st.success(f"Total chunks from all documents: {len(all_chunks)}")

    # --------------------------------------------------
    # Embeddings
    # --------------------------------------------------

    with st.spinner("Generating embeddings..."):

        embeddings = model.encode(all_chunks)

    st.success("Embeddings generated")

    # --------------------------------------------------
    # Query
    # --------------------------------------------------

    st.subheader("Ask a question about the documents")

    query = st.text_input("Enter your question")

    example_questions = [
        "What is the main topic of these documents?",
        "Summarize the important ideas",
        "What conclusions are presented?"
    ]

    example = st.selectbox("Example question:", [""] + example_questions)

    if example:
        query = example

    if query:

        with st.spinner("Running semantic retrieval..."):

            query_embedding = model.encode(query)

            similarities = np.array([
                cosine_similarity(query_embedding, emb)
                for emb in embeddings
            ])

            top_indices = similarities.argsort()[-top_k:][::-1]

        top_score = similarities[top_indices[0]]

        # --------------------------------------------------
        # Confidence
        # --------------------------------------------------

        if top_score > 0.45:
            st.success("High confidence retrieval")
        elif top_score > 0.25:
            st.warning("Medium confidence retrieval")
        else:
            st.error("Low confidence: information may not exist")

        # --------------------------------------------------
        # Query Statistics
        # --------------------------------------------------

        st.subheader("Query Statistics")

        c1, c2 = st.columns(2)

        c1.metric("Query Words", len(query.split()))
        c2.metric("Top Similarity", f"{top_score:.4f}")

        # --------------------------------------------------
        # Retrieved Chunks
        # --------------------------------------------------

        st.subheader("Retrieved Evidence")

        retrieved_chunks = []

        for rank, idx in enumerate(top_indices, start=1):

            source = chunk_sources[idx]

            with st.expander(
                f"Top {rank} | Doc: {source} | Chunk {idx} | Score {similarities[idx]:.4f}"
            ):

                highlighted = highlight_query_sentences(all_chunks[idx], query)

                st.markdown(highlighted)

                retrieved_chunks.append(
                    f"Doc: {source} | Chunk {idx} | Score {similarities[idx]:.4f}\n{all_chunks[idx]}"
                )

        # --------------------------------------------------
        # Similarity Distribution
        # --------------------------------------------------

        st.subheader("Similarity Distribution")

        st.line_chart(similarities)

        # --------------------------------------------------
        # Query-to-Chunk Attention Map
        # --------------------------------------------------

        st.subheader("Query-to-Chunk Attention Map")

        attention_df = pd.DataFrame({
            "Chunk": range(len(similarities)),
            "Similarity": similarities
        })

        st.bar_chart(attention_df.set_index("Chunk"))

        # --------------------------------------------------
        # Chunk Clustering Visualization
        # --------------------------------------------------

        st.subheader("Semantic Chunk Clustering")

        pca = PCA(n_components=2)

        reduced = pca.fit_transform(embeddings)

        cluster_df = pd.DataFrame({
            "x": reduced[:,0],
            "y": reduced[:,1],
            "Chunk": range(len(all_chunks))
        })

        st.scatter_chart(cluster_df, x="x", y="y")

        # --------------------------------------------------
        # Debug Dashboard
        # --------------------------------------------------

        st.subheader("RAG Debug Dashboard")

        d1, d2, d3 = st.columns(3)

        d1.metric("Total Chunks", len(all_chunks))
        d2.metric("Embedding Dimension", embeddings.shape[1])
        d3.metric("Top Similarity", f"{top_score:.4f}")

        d4, d5 = st.columns(2)

        d4.metric("Average Similarity", f"{similarities.mean():.4f}")
        d5.metric("Lowest Similarity", f"{similarities.min():.4f}")

        # --------------------------------------------------
        # Transparency
        # --------------------------------------------------

        st.subheader("Retrieved Chunk Index")

        for rank, idx in enumerate(top_indices, start=1):

            st.write(
                f"Rank {rank} | Document: {chunk_sources[idx]} | Chunk {idx} | Score {similarities[idx]:.4f}"
            )

        # --------------------------------------------------
        # Export
        # --------------------------------------------------

        if st.button("Export Retrieved Chunks"):

            export_text = "\n\n".join(retrieved_chunks)

            st.download_button(
                "Download Results",
                export_text,
                "retrieved_chunks.txt",
                "text/plain"
            )


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()

st.caption("Explain Yourself RAG — Advanced Retrieval Inspection Tool")