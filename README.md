# RAG Chatbot Using PDF and GPT Integration

This repository provides a **Retrieval-Augmented Generation (RAG)** chatbot application built with **Streamlit**, **FAISS**, **SentenceTransformers**, and **OpenAI GPT models**. The chatbot retrieves relevant information from a PDF document and generates natural language answers based on the retrieved content.

---

## Features

- Extracts and preprocesses text data from a PDF file.
- Generates or loads embeddings using **SentenceTransformers**.
- Creates or loads a **FAISS index** for fast similarity-based document retrieval.
- Integrates **OpenAI GPT** to generate responses based on retrieved content.
- Provides a simple and interactive **Streamlit** interface for user interaction.

---

## Prerequisites

Before running the application, ensure you have the following:

- **Python 3.8 or higher**
- Required Python libraries (listed in `requirements.txt`)
- OpenAI API key

---

## Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name
