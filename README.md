#RAG Chatbot

This repository provides a Retrieval-Augmented Generation (RAG) chatbot application built with Streamlit, FAISS, SentenceTransformers, and OpenAI GPT models. The chatbot retrieves relevant information from a PDF document and generates natural language answers based on the retrieved content.

Features
Extracts and preprocesses text data from a PDF file.
Generates or loads embeddings using SentenceTransformers.
Creates or loads a FAISS index for fast similarity-based document retrieval.
Integrates OpenAI GPT to generate responses based on retrieved content.
Provides a simple and interactive Streamlit interface for user interaction.
Prerequisites
Before running the application, ensure you have the following installed:

Python 3.8 or higher
Required Python libraries (listed in requirements.txt)
OpenAI API key
Installation
Clone this repository:

bash
Copy code
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
Install dependencies:

bash
Copy code
pip install -r requirements.txt
Add your OpenAI API key to the Streamlit secrets file:

Create a .streamlit/secrets.toml file and add the following:

toml
Copy code
[secrets]
OPENAI_API_KEY = "your-openai-api-key"
Place your PDF file in the data directory and rename it to Resume.pdf (or update the PDF_FILE path in the script).

Usage
Run the Streamlit application:

bash
Copy code
streamlit run app.py
Open your browser and navigate to the URL provided by Streamlit (default: http://localhost:8501).

Enter your query in the text input box and click Get Answer. The chatbot will retrieve relevant information from the PDF and generate a response.

File Structure
graphql
Copy code
├── app.py                 # Main application script
├── data/                  # Directory for storing data files
│   ├── Resume.pdf         # Input PDF file
│   ├── embeddings.npy     # Precomputed embeddings (generated after the first run)
│   └── faiss_index.bin    # FAISS index (generated after the first run)
├── requirements.txt       # List of required Python libraries
└── .streamlit/            # Directory for Streamlit configuration
    └── secrets.toml       # Streamlit secrets file (contains the OpenAI API key)
How It Works
Load Data: Text data is extracted from the PDF using PyPDF2 and stored in a DataFrame.
Generate/Load Embeddings: Sentence embeddings are generated for each row in the DataFrame using SentenceTransformer. If embeddings already exist, they are loaded from the embeddings.npy file.
Create/Load FAISS Index: A FAISS index is created for fast similarity-based search. If the index exists, it is loaded from the faiss_index.bin file.
Retrieve Documents: Given a user query, the top-k most similar rows are retrieved using the FAISS index.
Generate Response: The retrieved context is passed to the OpenAI GPT model to generate a natural language response.
Streamlit UI: A user-friendly interface allows users to interact with the chatbot.
Customization
PDF File: Replace Resume.pdf in the data directory with your own PDF file.
Model: The SentenceTransformer model can be replaced with another pre-trained model.
Query Response: Modify the GPT prompt in the generate_response function to adjust the style and tone of the responses.
Known Issues
Ensure that the PDF text is extractable (e.g., not scanned images). Use OCR tools like Tesseract for PDFs containing images.
Large PDF files may increase processing time for embedding generation.
