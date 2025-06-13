import streamlit as st
import torch
import transformers
from torch import cuda, bfloat16
from transformers import AutoTokenizer
from time import time
from langchain.llms import HuggingFacePipeline
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
import config

def load_model():
    """Load the model and create pipeline"""
    print("Loading model...")
    
    device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Quantization config
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=bfloat16
    )
    
    # Load model
    model_config = transformers.AutoConfig.from_pretrained(config.MODEL_NAME)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        config.MODEL_NAME,
        trust_remote_code=True,
        config=model_config,
        quantization_config=bnb_config,
        device_map='auto',
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(config.MODEL_NAME)
    
    # Create pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    
    # Create LLM
    llm = HuggingFacePipeline(pipeline=pipeline)
    
    print("Model loaded successfully!")
    return llm, tokenizer, model

def process_pdf(pdf_path):
    """Load and split PDF document"""
    print(f"Processing PDF: {pdf_path}")
    
    # Load PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    
    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
        strip_whitespace=True
    )
    
    splits = text_splitter.split_documents(documents)
    print(f"Created {len(splits)} document chunks")
    
    return splits

def create_vector_store(document_splits):
    """Create embeddings and vector store"""
    print("Creating embeddings and vector store...")
    
    # Create embeddings
    device = "cuda" if cuda.is_available() else "cpu"
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        model_kwargs={"device": device}
    )
    
    # Create vector store
    vectordb = Chroma.from_documents(
        documents=document_splits,
        embedding=embeddings,
        persist_directory="chroma_db"
    )
    
    # Create retriever
    retriever = vectordb.as_retriever()
    
    print("Vector store created!")
    return retriever

def create_qa_chain(llm, retriever):
    """Create the QA chain"""
    print("Creating QA chain...")
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True
    )
    
    print("QA chain ready!")
    return qa

def ask_question(qa_chain, question):
    """Ask a question and get answer"""
    print(f"\nQuestion: {question}")
    print("-" * 50)
    
    start_time = time()
    result = qa_chain.run(question)
    end_time = time()
    
    print(f"\nAnswer: {result}")
    print(f"\nTime taken: {round(end_time - start_time, 2)} seconds")
    print("=" * 50)
    
    return result

def main():
    try:
        # Step 1: Load model
        llm, tokenizer, model = load_model()
        
        # Step 2: Process PDF
        document_splits = process_pdf(config.PDF_PATH)
        
        # Step 3: Create vector store
        retriever = create_vector_store(document_splits)
        
        # Step 4: Create QA chain
        qa_chain = create_qa_chain(llm, retriever)
        
        # Step 5: Ask questions
        print("\n" + "="*50)
        print("RAG SYSTEM READY!")
        print("="*50)
        
        # Default question from your notebook
        ask_question(qa_chain, "Advantage of CCT over ViT")
        
        # Interactive mode
        print("\nYou can now ask questions! (type 'quit' to exit)")
        while True:
            question = input("\nYour question: ").strip()
            
            if question.lower() in ['quit', 'exit', 'q']:
                break
                
            if question:
                ask_question(qa_chain, question)
        
        # Optional: Save model
        save_choice = input("\nDo you want to save the model? (y/n): ")
        if save_choice.lower() == 'y':
            print("Saving model...")
            tokenizer.save_pretrained("./saved_model/")
            model.save_pretrained("./saved_model/")
            print("Model saved to ./saved_model/")
        
        print("\nThank you for using the RAG system!")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure your PDF path in config.py is correct!")

if __name__ == "__main__":
    main()
