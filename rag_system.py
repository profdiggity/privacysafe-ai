#!/usr/bin/env python3
"""
Robust RAG System with Ollama Gemma 3B
Automatically processes PDFs and provides precise answers
"""

import os
import time
import hashlib
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import re

# Core dependencies
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_core.documents import Document

# File monitoring
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

@dataclass
class RAGConfig:
    """Configuration for the RAG system"""
    pdf_folder: str = "./documents"
    vector_db_path: str = "./vector_db"
    model_name: str = "gemma3:4b"
    embedding_model: str = "nomic-embed-text:latest"
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_retrieval_docs: int = 4
    temperature: float = 0.0
    top_p: float = 0.1

class DocumentProcessor:
    """Handles PDF processing and text extraction"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def process_pdf(self, pdf_path: str) -> List[Document]:
        """Process a single PDF file"""
        try:
            loader = PyPDFLoader(pdf_path)
            documents = loader.load()

            # Add metadata
            for doc in documents:
                doc.metadata.update({
                    'source_file': os.path.basename(pdf_path),
                    'processed_at': datetime.now().isoformat(),
                    'file_hash': self._get_file_hash(pdf_path)
                })

            # Split documents
            split_docs = self.text_splitter.split_documents(documents)
            logging.info(f"Processed {pdf_path}: {len(split_docs)} chunks")
            return split_docs

        except Exception as e:
            logging.error(f"Error processing {pdf_path}: {str(e)}")
            return []

    def _get_file_hash(self, filepath: str) -> str:
        """Generate hash for file to detect changes"""
        hash_md5 = hashlib.md5()
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

class VectorStoreManager:
    """Manages vector database operations"""

    def __init__(self, config: RAGConfig):
        self.config = config
        self.embeddings = OllamaEmbeddings(model=config.embedding_model)
        self.vector_store = None
        self.processed_files = {}
        self._load_processed_files()

    def _load_processed_files(self):
        """Load record of processed files"""
        processed_files_path = os.path.join(self.config.vector_db_path, "processed_files.pkl")
        if os.path.exists(processed_files_path):
            with open(processed_files_path, 'rb') as f:
                self.processed_files = pickle.load(f)

    def _save_processed_files(self):
        """Save record of processed files"""
        os.makedirs(self.config.vector_db_path, exist_ok=True)
        processed_files_path = os.path.join(self.config.vector_db_path, "processed_files.pkl")
        with open(processed_files_path, 'wb') as f:
            pickle.dump(self.processed_files, f)

    def load_vector_store(self):
        """Load existing vector store"""
        vector_store_path = os.path.join(self.config.vector_db_path, "faiss_index")
        if os.path.exists(vector_store_path):
            try:
                self.vector_store = FAISS.load_local(
                    vector_store_path,
                    self.embeddings,
                    allow_dangerous_deserialization=True
                )
                logging.info("Loaded existing vector store")
            except Exception as e:
                logging.error(f"Error loading vector store: {str(e)}")
                self.vector_store = None

    def save_vector_store(self):
        """Save vector store to disk"""
        if self.vector_store:
            os.makedirs(self.config.vector_db_path, exist_ok=True)
            vector_store_path = os.path.join(self.config.vector_db_path, "faiss_index")
            self.vector_store.save_local(vector_store_path)
            self._save_processed_files()
            logging.info("Saved vector store")

    def add_documents(self, documents: List[Document], file_path: str, file_hash: str):
        """Add documents to vector store"""
        if not documents:
            return

        try:
            if self.vector_store is None:
                self.vector_store = FAISS.from_documents(documents, self.embeddings)
            else:
                self.vector_store.add_documents(documents)

            # Record the processed file
            self.processed_files[file_path] = {
                'hash': file_hash,
                'processed_at': datetime.now().isoformat(),
                'num_chunks': len(documents)
            }

            self.save_vector_store()
            logging.info(f"Added {len(documents)} documents from {file_path}")

        except Exception as e:
            logging.error(f"Error adding documents: {str(e)}")

    def get_retriever(self):
        """Get retriever for similarity search"""
        if self.vector_store:
            return self.vector_store.as_retriever(
                search_kwargs={"k": self.config.max_retrieval_docs}
            )
        return None

    def needs_processing(self, file_path: str, file_hash: str) -> bool:
        """Check if file needs processing"""
        if file_path not in self.processed_files:
            return True
        return self.processed_files[file_path]['hash'] != file_hash

class ResponseFilter:
    """Filters and cleans model responses"""

    @staticmethod
    def clean_response(response: str) -> str:
        """Clean and filter response to ensure precision"""
        response = response.strip()

        # Patterns that indicate unwanted explanatory language
        unwanted_patterns = [
            r"according to.*?(?:text|context|document|provided)",
            r"the.*?(?:text|context|document).*?(?:doesn't|does not|mentions|states)",
            r"based on.*?(?:context|information|document)",
            r"(?:this|the).*?(?:text|document|article).*?(?:doesn't|does not)",
            r"(?:from|in).*?(?:the|this).*?(?:text|document|context)",
            r"according to.*?(?:provided|given|above)",
            r"the information.*?(?:provided|given|states)",
            r"(?:i can see|i found|i notice).*?(?:that|from)",
        ]

        # Check for unwanted patterns
        for pattern in unwanted_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                return "I don't know the answer to that question."

        # If response is too verbose (likely explanatory)
        if len(response) > 150 and any(word in response.lower() for word in
                                      ["according", "provided", "context", "text", "document", "based on"]):
            return "I don't know the answer to that question."

        # Remove common prefixes
        prefixes_to_remove = [
            "according to the text, ",
            "according to the context, ",
            "based on the provided context, ",
            "the text states that ",
            "the context mentions that ",
        ]

        for prefix in prefixes_to_remove:
            if response.lower().startswith(prefix):
                response = response[len(prefix):]
                break

        return response.strip()

class PDFFileHandler(FileSystemEventHandler):
    """Handles file system events for PDF monitoring"""

    def __init__(self, rag_system):
        self.rag_system = rag_system

    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith('.pdf'):
            logging.info(f"New PDF detected: {event.src_path}")
            time.sleep(1)  # Wait for file to be completely written
            self.rag_system.process_single_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory and event.src_path.endswith('.pdf'):
            logging.info(f"PDF modified: {event.src_path}")
            time.sleep(1)  # Wait for file to be completely written
            self.rag_system.process_single_file(event.src_path)

class RobustRAGSystem:
    """Main RAG system class"""

    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.setup_logging()
        self.setup_directories()

        # Initialize components
        self.doc_processor = DocumentProcessor(self.config)
        self.vector_manager = VectorStoreManager(self.config)
        self.response_filter = ResponseFilter()

        # Initialize model
        self.model = OllamaLLM(
            model=self.config.model_name,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            repeat_penalty=1.1
        )

        # File monitoring
        self.observer = None

        # Load existing vector store
        self.vector_manager.load_vector_store()

        # Process existing files
        self.process_existing_files()

        # Setup query chain
        self.setup_query_chain()

    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('rag_system.log'),
                logging.StreamHandler()
            ]
        )

    def setup_directories(self):
        """Create necessary directories"""
        os.makedirs(self.config.pdf_folder, exist_ok=True)
        os.makedirs(self.config.vector_db_path, exist_ok=True)

    def setup_query_chain(self):
        """Setup the query processing chain"""
        template = """You are a precise fact-checking system. Follow these rules EXACTLY:

1. Only answer if the EXACT information is in the context below
2. If the answer is not in the context, respond with: "I don't know the answer to that question."
3. Give only direct, factual answers - no explanations or commentary
4. Do not use phrases like "according to", "the text says", etc.
5. Keep answers brief and to the point

Context: {context}

Question: {question}

Answer:"""

        self.prompt = PromptTemplate.from_template(template)

        # Setup retrieval chain
        retriever = self.vector_manager.get_retriever()
        if retriever:
            self.retrieval_chain = RunnableParallel({
                "context": RunnableLambda(lambda x: self._format_context(retriever.invoke(x["question"]))),
                "question": RunnableLambda(lambda x: x["question"]),
            })

            self.query_chain = (
                self.retrieval_chain
                | self.prompt
                | self.model
                | RunnableLambda(self.response_filter.clean_response)
            )
        else:
            self.query_chain = None

    def _format_context(self, docs: List[Document]) -> str:
        """Format retrieved documents as context"""
        return "\n\n".join([doc.page_content for doc in docs])

    def process_existing_files(self):
        """Process all existing PDF files"""
        pdf_files = list(Path(self.config.pdf_folder).glob("*.pdf"))

        if not pdf_files:
            logging.info("No PDF files found in the documents folder")
            return

        logging.info(f"Found {len(pdf_files)} PDF files to process")

        for pdf_file in pdf_files:
            self.process_single_file(str(pdf_file))

    def process_single_file(self, file_path: str):
        """Process a single PDF file"""
        try:
            file_hash = self.doc_processor._get_file_hash(file_path)

            if not self.vector_manager.needs_processing(file_path, file_hash):
                logging.info(f"Skipping {file_path} - already processed")
                return

            logging.info(f"Processing {file_path}")
            documents = self.doc_processor.process_pdf(file_path)

            if documents:
                self.vector_manager.add_documents(documents, file_path, file_hash)
                # Refresh query chain with updated vector store
                self.setup_query_chain()
                logging.info(f"Successfully processed {file_path}")
            else:
                logging.warning(f"No documents extracted from {file_path}")

        except Exception as e:
            logging.error(f"Error processing {file_path}: {str(e)}")

    def start_monitoring(self):
        """Start monitoring the PDF folder for new files"""
        if self.observer:
            self.stop_monitoring()

        event_handler = PDFFileHandler(self)
        self.observer = Observer()
        self.observer.schedule(event_handler, self.config.pdf_folder, recursive=False)
        self.observer.start()
        logging.info(f"Started monitoring {self.config.pdf_folder} for new PDFs")

    def stop_monitoring(self):
        """Stop monitoring the PDF folder"""
        if self.observer:
            self.observer.stop()
            self.observer.join()
            self.observer = None
            logging.info("Stopped monitoring for new PDFs")

    def query(self, question: str) -> str:
        """Query the RAG system"""
        if not self.query_chain:
            return "No documents have been processed yet. Please add PDF files to the documents folder."

        try:
            response = self.query_chain.invoke({"question": question})
            return response
        except Exception as e:
            logging.error(f"Error processing query: {str(e)}")
            return "An error occurred while processing your question."

    def get_system_info(self) -> Dict[str, Any]:
        """Get system information"""
        return {
            "pdf_folder": self.config.pdf_folder,
            "vector_db_path": self.config.vector_db_path,
            "processed_files": len(self.vector_manager.processed_files),
            "model": self.config.model_name,
            "embedding_model": self.config.embedding_model,
            "monitoring_active": self.observer is not None
        }

def main():
    """Main function to run the RAG system"""
    print("🚀 Initializing Robust RAG System...")

    # Create custom config if needed
    config = RAGConfig(
        pdf_folder="./documents",
        vector_db_path="./vector_db",
        model_name="gemma3:4b",
        embedding_model="nomic-embed-text:latest"
    )

    # Initialize RAG system
    rag = RobustRAGSystem(config)

    # Start monitoring for new files
    rag.start_monitoring()

    # Print system info
    info = rag.get_system_info()
    print("\n📊 System Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    print(f"\n📁 Drop PDF files into: {config.pdf_folder}")
    print("💬 Ready to answer questions!\n")

    # Interactive query loop
    try:
        while True:
            question = input("\n🤔 Ask a question (or 'quit' to exit): ").strip()

            if question.lower() in ['quit', 'exit', 'q']:
                break

            if not question:
                continue

            print("🔍 Searching...")
            answer = rag.query(question)
            print(f"📝 Answer: {answer}")

    except KeyboardInterrupt:
        print("\n\n👋 Shutting down...")

    finally:
        rag.stop_monitoring()
        print("✅ RAG system stopped.")

if __name__ == "__main__":
    main()