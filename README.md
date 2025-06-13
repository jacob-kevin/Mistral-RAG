# Mistral-RAG

A simple RAG (Retrieval-Augmented Generation) system that answers questions about your PDF documents using Mistral-7B model.

## Quick Setup

1. **Install requirements:**
```bash
pip install -r requirements.txt
```

2. **Put your PDF file in the data folder:**
```bash
mkdir data
# Copy your PDF file to data/
```

3. **Update config.py:**
```python
PDF_PATH = "data/your_document.pdf"  # Change to your PDF filename
```

4. **Run the program:**
```bash
python main.py
```

## That's it! 

The program will:
1. Load the AI model
2. Process your PDF
3. Create a searchable database
4. Answer questions about your document

## Usage

- The program starts with a default question
- Then you can ask your own questions
- Type 'quit' to exit

## Example Questions

- "What is the main topic of this document?"
- "Summarize the key findings"
- "What are the advantages mentioned?"

## Requirements

- Python 3.9+
- NVIDIA GPU (recommended)
- 8GB+ GPU memory

## Troubleshooting

**Error loading model?**
- Make sure you have a good internet connection (first time downloads the model)
- Check if you have enough GPU memory

**Can't find PDF?**
- Make sure your PDF path in config.py is correct
- Check that the PDF file exists in the data folder

**Out of memory?**
- Close other programs using GPU
- Try with a smaller PDF file first
