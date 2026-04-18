```markdown
# Multimodal RAG System for Document QA 📄🔍

This repository contains a local, **Multimodal Retrieval-Augmented Generation (RAG)** application built for answering questions based on PDF documents. 

Unlike traditional RAG systems that rely on Optical Character Recognition (OCR) to extract text (which often breaks the formatting of tables and charts), this system treats document pages as complete high-resolution images. It uses a Vision-Language Model (VLM) to "read" the pages, preserving the spatial layout, figures, and context perfectly.

This project was developed for a university Multimedia assignment and is fully optimized to run on a **CPU-only** environment without requiring a dedicated NVIDIA GPU.

## 🌟 Key Features
* **Visual Document Indexing:** Uses `Byaldi` and the **ColPali-v1.2** model to create embeddings directly from PDF page images.
* **Smart Reasoning:** Employs **Qwen2-VL-2B-Instruct** as the core reasoning engine to analyze retrieved images and generate accurate answers.
* **CPU Optimized:** Deep-level environment variables configured to bypass CUDA/GPU dependencies and run reliably on System RAM and CPU.
* **Local Index Caching:** Automatically saves the heavy vector embeddings locally (`.byaldi` folder) to reduce future startup times from 30+ minutes down to a few seconds.
* **Interactive UI:** Features a sleek, real-time web interface built with **Gradio**.

## 🛠️ Tech Stack
* **Python 3.11**
* **Byaldi / ColPali** (Indexing & Retrieval)
* **Transformers / Hugging Face** (Vision-Language Model)
* **Gradio** (User Interface)
* **pdf2image & Poppler** (PDF processing)

## 🚀 Installation & Setup

### 1. Clone the repository
```bash
git clone [https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git](https://github.com/YOUR-USERNAME/YOUR-REPO-NAME.git)
cd YOUR-REPO-NAME
```

### 2. Set up the virtual environment
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Poppler (Required for PDF conversion)
* **Windows:** Download [Poppler for Windows](https://github.com/oschwartz10612/poppler-windows/releases/), extract it, and place the `poppler` folder directly into the root directory of this project. The Python code automatically links to `poppler\Library\bin` via `os.environ["PATH"]`.
* **Linux (Colab/Ubuntu):** Run `sudo apt-get install -y poppler-utils`.
* **Mac:** Run `brew install poppler`.

## 💻 Usage

To start the application, run the following command from the root of the project:

```bash
python -m src.app
```

**Note on First Run:** If there is no cached index in the `.byaldi` folder, the application will begin indexing the raw PDF. Because this system is optimized for CPU execution, converting and embedding a large PDF (e.g., 300 pages) may take **15–30 minutes**. Please do not close the terminal during this phase! 

Once finished, the index is saved locally, and all future runs will load in just a few seconds. A local URL (e.g., `http://127.0.0.1:7860`) will be generated for you to access the Gradio UI.

## 📂 Project Structure

```text
├── .byaldi/                 # Hidden folder containing the saved vector index (git-ignored)
├── data/
│   └── raw/                 # Contains the source PDF(s) to be processed
├── poppler/                 # Poppler binaries for Windows PDF processing
├── src/
│   ├── app.py               # Main execution file and Gradio UI setup
│   ├── data_processing.py   # PDF to Image conversion and ColPali indexing logic
│   └── model.py             # Qwen VLM loading and inference logic
├── .gitignore               # Ignored files (venv, caches, etc.)
├── README.md                # Project documentation
└── requirements.txt         # Python package dependencies
```
```
