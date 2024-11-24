import os
import pandas as pd
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import streamlit as st
import openai
from PyPDF2 import PdfReader

# --- CONFIGURATION ---
PDF_FILE = "data/manual.pdf"
EMBEDDINGS_FILE = "data/embeddings.npy"
FAISS_INDEX_FILE = "data/faiss_index.bin"

# Set up OpenAI API key
openai.api_key = st.secrets["OPENAI_API_KEY"]

# --- LOAD DATA FROM PDF ---
@st.cache_data
def load_data_from_pdf(file_path):
    """Load and preprocess text data from a PDF file."""
    reader = PdfReader(file_path)
    text_data = []
    for page in reader.pages:
        text_data.append(page.extract_text())
    return pd.DataFrame({"text": text_data})  # Convert to a DataFrame for uniform processing

data = load_data_from_pdf(PDF_FILE)

# --- CONCATENATE COLUMNS (ADAPTED FOR PDF TEXT) ---
def concatenate_row(row):
    """
    Concatenate all columns in a row into a single string.
    Handles text and numeric columns by converting them to strings.
    """
    return " ".join(row.astype(str).values)

data['concatenated'] = data.apply(concatenate_row, axis=1)

# --- GENERATE OR LOAD EMBEDDINGS ---
@st.cache_resource
def load_or_generate_embeddings(data, column_name):
    """Load or generate embeddings for the concatenated rows."""
    model = SentenceTransformer('msmarco-distilbert-base-v4')  # Use a search-optimized model
    if os.path.exists(EMBEDDINGS_FILE):
        embeddings = np.load(EMBEDDINGS_FILE)
    else:
        embeddings = model.encode(data[column_name].tolist())
        np.save(EMBEDDINGS_FILE, embeddings)  # Save embeddings to disk
    return model, embeddings

model, embeddings = load_or_generate_embeddings(data, 'concatenated')

# --- CREATE OR LOAD FAISS INDEX ---
@st.cache_resource
def load_or_create_faiss_index(embeddings):
    """Load or create a FAISS index for fast similarity search."""
    index = faiss.IndexFlatL2(embeddings.shape[1])
    if os.path.exists(FAISS_INDEX_FILE):
        index = faiss.read_index(FAISS_INDEX_FILE)  # Fixed FAISS index loading
    else:
        index.add(embeddings)
        faiss.write_index(index, FAISS_INDEX_FILE)  # Save the index to disk
    return index

faiss_index = load_or_create_faiss_index(np.array(embeddings))

# --- RETRIEVAL FUNCTION ---
def retrieve_documents(query, k=5):
    """Retrieve the top-k most similar rows for a given query."""
    query_embedding = model.encode([query])[0]
    distances, indices = faiss_index.search(np.array([query_embedding]), k)

    # Filter retrieved rows to prioritize relevance
    results = data.iloc[indices[0]]
    filtered_results = results[results['concatenated'].str.contains(query.split()[-1], case=False, na=False)]

    if not filtered_results.empty:
        return filtered_results
    return results  # Return original results if no match is found

# --- GPT-4 RESPONSE GENERATION ---
def generate_response(query):
    """Generate a response using GPT-4 and retrieved documents."""
    retrieved_docs = retrieve_documents(query)
    context = "\n".join(retrieved_docs['concatenated'].tolist())
    prompt = f"Context: {context}\n\nUser Query: {query}\n\nResponse:"
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": "You are a helpful assistant. Use the retrieved embeddings to answer user queries and also make it easy understandable for users."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200,
            temperature=0
        )
        return response['choices'][0]['message']['content'].strip()
    except Exception as e:
        return f"Error: {str(e)}"

# --- ADD BACKGROUND IMAGE WITH OVERLAY ---
# --- ADD BACKGROUND IMAGE WITH TEXT STYLING ---
def add_background_image_with_text_styling():
    background_image_url = "https://www.ford.com/is/image/content/dam/vdm_ford/live/en_us/ford/nameplate/mustang/2024/collections/dm/24_FRD_MST_61047.tif?croppathe=1_3x2&wid=1440"  # Replace with your Mustang image URL
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: url("{background_image_url}") no-repeat center center fixed;
            background-size: cover;
        }}
        .content-box {{
            background-color: rgba(255, 255, 255, 0.9);  /* Semi-transparent white background */
            border-radius: 10px;
            padding: 20px;
            max-width: 800px;
            margin: 50px auto;  /* Center the box with spacing */
            box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.2); /* Add shadow for visibility */
        }}
        h1, h2, h3, h4, h5, h6 {{
            color: #1f77b4;  /* Set header text color */
        }}
        p {{
            color: #ff4500;  /* Set paragraph text color */
            background-color: #ffff99;  /* Highlight text with yellow */
            padding: 5px;  /* Add padding to make the highlight more noticeable */
            border-radius: 5px;  /* Smooth corners for highlighted text */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
    st.markdown('<div class="content-box">', unsafe_allow_html=True)

# Call the function to add the background and text styles
add_background_image_with_text_styling()

# --- STREAMLIT UI ---
st.title("RAG Chatbot")
st.write("Ask a question, and the chatbot will provide answers based on the PDF content.")

user_query = st.text_input("Enter your question:")

if st.button("Get Answer"):
    if user_query:
        with st.spinner("Retrieving documents and generating response..."):
            response = generate_response(user_query)
        st.subheader("Response:")
        st.write(response)  # The response text will now follow the new text styling
    else:
        st.warning("Please enter a question.")

st.markdown('</div>', unsafe_allow_html=True)
