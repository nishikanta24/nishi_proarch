"""
Embeddings Module
Handles document loading, chunking, embedding generation, and vector store management.
Uses LangChain for document processing and FAISS for vector storage.
"""

import os
import logging
from typing import List, Optional, Dict
from pathlib import Path
import pickle

from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.schema import Document

from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


class EmbeddingsManager:
    """
    Manages document embeddings and vector store operations.
    Handles loading reference documents, generating embeddings, and FAISS vector store.
    """
    
    def __init__(
        self, 
        reference_docs_path: str = "data/reference",
        vector_store_path: str = "data/vector_store",
        request_id: str = "default"
    ):
        """
        Initialize the embeddings manager
        
        Args:
            reference_docs_path: Path to reference documents directory
            vector_store_path: Path to save/load vector store
            request_id: Request ID for logging
        """
        self.reference_docs_path = reference_docs_path
        self.vector_store_path = vector_store_path
        self.request_id = request_id
        self.vector_store = None
        self.embeddings = None
        
        logger.info(
            f"Initializing EmbeddingsManager with docs path: {reference_docs_path}",
            extra={'request_id': request_id}
        )
        
        # Create directories if they don't exist
        os.makedirs(vector_store_path, exist_ok=True)
        os.makedirs(reference_docs_path, exist_ok=True)
        
        # Initialize embeddings model
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embeddings model (using HuggingFace for free, local embeddings)"""
        try:
            logger.info(
                "Initializing HuggingFace embeddings model...",
                extra={'request_id': self.request_id}
            )
            
            # Using a lightweight, fast, and accurate model
            # all-MiniLM-L6-v2 is small (80MB), fast, and good quality
            self.embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            logger.info(
                "✓ Embeddings model initialized successfully",
                extra={'request_id': self.request_id}
            )
            
        except Exception as e:
            logger.error(
                f"Failed to initialize embeddings model: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def load_reference_documents(self) -> List[Document]:
        """
        Load all reference documents from the reference directory
        
        Returns:
            List of LangChain Document objects
        """
        logger.info(
            f"Loading reference documents from {self.reference_docs_path}",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Check if directory exists and has files
            ref_path = Path(self.reference_docs_path)
            if not ref_path.exists():
                logger.warning(
                    f"Reference docs path does not exist: {self.reference_docs_path}",
                    extra={'request_id': self.request_id}
                )
                return []
            
            # Get all text files
            txt_files = list(ref_path.glob("*.txt"))
            md_files = list(ref_path.glob("*.md"))
            all_files = txt_files + md_files
            
            if not all_files:
                logger.warning(
                    f"No .txt or .md files found in {self.reference_docs_path}",
                    extra={'request_id': self.request_id}
                )
                return []
            
            logger.info(
                f"Found {len(all_files)} document files to load",
                extra={'request_id': self.request_id}
            )
            
            # Load documents
            documents = []
            for file_path in all_files:
                try:
                    logger.info(
                        f"Loading document: {file_path.name}",
                        extra={'request_id': self.request_id}
                    )
                    
                    loader = TextLoader(str(file_path), encoding='utf-8')
                    docs = loader.load()
                    
                    # Add metadata
                    for doc in docs:
                        doc.metadata['source'] = file_path.name
                        doc.metadata['type'] = 'reference_document'
                    
                    documents.extend(docs)
                    
                    logger.info(
                        f"✓ Loaded {len(docs)} chunks from {file_path.name}",
                        extra={'request_id': self.request_id}
                    )
                    
                except Exception as e:
                    logger.error(
                        f"Failed to load {file_path.name}: {str(e)}",
                        extra={'request_id': self.request_id}
                    )
                    continue
            
            logger.info(
                f"✓ Total documents loaded: {len(documents)}",
                extra={'request_id': self.request_id}
            )
            
            return documents
            
        except Exception as e:
            logger.error(
                f"Error loading reference documents: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def chunk_documents(
        self, 
        documents: List[Document],
        chunk_size: int = 1000,
        chunk_overlap: int = 200
    ) -> List[Document]:
        """
        Split documents into smaller chunks for better retrieval
        
        Args:
            documents: List of documents to chunk
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
            
        Returns:
            List of chunked documents
        """
        logger.info(
            f"Chunking {len(documents)} documents (chunk_size={chunk_size}, overlap={chunk_overlap})",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Use RecursiveCharacterTextSplitter for intelligent chunking
            # It tries to split on paragraphs, then sentences, then words
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                length_function=len,
                separators=["\n\n", "\n", ". ", " ", ""]
            )
            
            chunked_docs = text_splitter.split_documents(documents)
            
            logger.info(
                f"✓ Created {len(chunked_docs)} chunks from {len(documents)} documents",
                extra={'request_id': self.request_id}
            )
            
            # Log sample chunk for verification
            if chunked_docs:
                sample_chunk = chunked_docs[0]
                logger.info(
                    f"Sample chunk length: {len(sample_chunk.page_content)} chars, "
                    f"source: {sample_chunk.metadata.get('source', 'unknown')}",
                    extra={'request_id': self.request_id}
                )
            
            return chunked_docs
            
        except Exception as e:
            logger.error(
                f"Error chunking documents: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def create_vector_store(
        self, 
        documents: List[Document],
        force_recreate: bool = False
    ) -> FAISS:
        """
        Create or load FAISS vector store with document embeddings
        
        Args:
            documents: List of documents to embed
            force_recreate: Force recreation even if existing store found
            
        Returns:
            FAISS vector store
        """
        vector_store_file = os.path.join(self.vector_store_path, "faiss_index")
        
        # Check if vector store already exists
        if os.path.exists(f"{vector_store_file}.faiss") and not force_recreate:
            logger.info(
                "Found existing vector store, loading...",
                extra={'request_id': self.request_id}
            )
            return self.load_vector_store()
        
        logger.info(
            f"Creating new vector store with {len(documents)} documents",
            extra={'request_id': self.request_id}
        )
        
        try:
            if not documents:
                raise ValueError("No documents provided for vector store creation")
            
            # Create FAISS vector store
            logger.info(
                "Generating embeddings and creating FAISS index...",
                extra={'request_id': self.request_id}
            )
            
            self.vector_store = FAISS.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
            
            # Save vector store
            self.save_vector_store()
            
            logger.info(
                f"✓ Vector store created successfully with {len(documents)} documents",
                extra={'request_id': self.request_id}
            )
            
            return self.vector_store
            
        except Exception as e:
            logger.error(
                f"Error creating vector store: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def save_vector_store(self):
        """Save vector store to disk"""
        if self.vector_store is None:
            logger.warning(
                "No vector store to save",
                extra={'request_id': self.request_id}
            )
            return
        
        try:
            vector_store_file = os.path.join(self.vector_store_path, "faiss_index")
            
            logger.info(
                f"Saving vector store to {vector_store_file}",
                extra={'request_id': self.request_id}
            )
            
            self.vector_store.save_local(vector_store_file)
            
            logger.info(
                "✓ Vector store saved successfully",
                extra={'request_id': self.request_id}
            )
            
        except Exception as e:
            logger.error(
                f"Error saving vector store: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def load_vector_store(self) -> FAISS:
        """
        Load existing vector store from disk
        
        Returns:
            FAISS vector store
        """
        try:
            vector_store_file = os.path.join(self.vector_store_path, "faiss_index")
            
            logger.info(
                f"Loading vector store from {vector_store_file}",
                extra={'request_id': self.request_id}
            )
            
            if not os.path.exists(f"{vector_store_file}.faiss"):
                raise FileNotFoundError(f"Vector store not found at {vector_store_file}")
            
            self.vector_store = FAISS.load_local(
                vector_store_file,
                self.embeddings,
                allow_dangerous_deserialization=True
            )
            
            logger.info(
                "✓ Vector store loaded successfully",
                extra={'request_id': self.request_id}
            )
            
            return self.vector_store
            
        except Exception as e:
            logger.error(
                f"Error loading vector store: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def add_documents_to_store(self, documents: List[Document]):
        """
        Add new documents to existing vector store
        
        Args:
            documents: List of documents to add
        """
        if self.vector_store is None:
            logger.warning(
                "No vector store loaded, creating new one",
                extra={'request_id': self.request_id}
            )
            self.vector_store = self.create_vector_store(documents)
            return
        
        try:
            logger.info(
                f"Adding {len(documents)} documents to existing vector store",
                extra={'request_id': self.request_id}
            )
            
            self.vector_store.add_documents(documents)
            self.save_vector_store()
            
            logger.info(
                "✓ Documents added successfully",
                extra={'request_id': self.request_id}
            )
            
        except Exception as e:
            logger.error(
                f"Error adding documents to vector store: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def similarity_search(
        self, 
        query: str, 
        k: int = 5,
        filter_metadata: Optional[Dict] = None
    ) -> List[Document]:
        """
        Search for similar documents in vector store
        
        Args:
            query: Search query
            k: Number of results to return
            filter_metadata: Optional metadata filter
            
        Returns:
            List of most similar documents
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_vector_store() first.")
        
        try:
            logger.info(
                f"Performing similarity search for query: '{query[:100]}...' (k={k})",
                extra={'request_id': self.request_id}
            )
            
            if filter_metadata:
                results = self.vector_store.similarity_search(
                    query, 
                    k=k,
                    filter=filter_metadata
                )
            else:
                results = self.vector_store.similarity_search(query, k=k)
            
            logger.info(
                f"✓ Found {len(results)} similar documents",
                extra={'request_id': self.request_id}
            )
            
            # Log retrieved sources
            sources = [doc.metadata.get('source', 'unknown') for doc in results]
            logger.info(
                f"Retrieved from sources: {list(set(sources))}",
                extra={'request_id': self.request_id}
            )
            
            return results
            
        except Exception as e:
            logger.error(
                f"Error in similarity search: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def similarity_search_with_score(
        self, 
        query: str, 
        k: int = 5
    ) -> List[tuple]:
        """
        Search with relevance scores
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of (Document, score) tuples
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized")
        
        try:
            logger.info(
                f"Performing similarity search with scores for query: '{query[:100]}...'",
                extra={'request_id': self.request_id}
            )
            
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            logger.info(
                f"✓ Found {len(results)} results with scores",
                extra={'request_id': self.request_id}
            )
            
            # Log scores
            for i, (doc, score) in enumerate(results[:3]):
                logger.info(
                    f"  Result {i+1}: score={score:.4f}, source={doc.metadata.get('source', 'unknown')}",
                    extra={'request_id': self.request_id}
                )
            
            return results
            
        except Exception as e:
            logger.error(
                f"Error in similarity search with score: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise
    
    def get_vector_store_stats(self) -> Dict:
        """
        Get statistics about the vector store
        
        Returns:
            Dictionary with vector store statistics
        """
        if self.vector_store is None:
            return {
                'status': 'not_initialized',
                'message': 'Vector store not loaded'
            }
        
        try:
            # Get number of vectors in FAISS index
            num_vectors = self.vector_store.index.ntotal
            
            stats = {
                'status': 'initialized',
                'total_vectors': num_vectors,
                'embedding_dimension': self.vector_store.index.d,
                'vector_store_path': self.vector_store_path
            }
            
            logger.info(
                f"Vector store stats: {num_vectors} vectors, dimension={self.vector_store.index.d}",
                extra={'request_id': self.request_id}
            )
            
            return stats
            
        except Exception as e:
            logger.error(
                f"Error getting vector store stats: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def rebuild_vector_store(self):
        """
        Rebuild vector store from scratch
        Useful when reference documents are updated
        """
        logger.info(
            "Rebuilding vector store from scratch",
            extra={'request_id': self.request_id}
        )
        
        try:
            # Load documents
            documents = self.load_reference_documents()
            
            if not documents:
                raise ValueError("No documents found to build vector store")
            
            # Chunk documents
            chunked_docs = self.chunk_documents(documents)
            
            # Create new vector store (force recreate)
            self.create_vector_store(chunked_docs, force_recreate=True)
            
            logger.info(
                "✓ Vector store rebuilt successfully",
                extra={'request_id': self.request_id}
            )
            
        except Exception as e:
            logger.error(
                f"Error rebuilding vector store: {str(e)}",
                extra={'request_id': self.request_id},
                exc_info=True
            )
            raise


def initialize_embeddings(
    reference_docs_path: str = "data/reference",
    vector_store_path: str = "data/vector_store",
    force_rebuild: bool = False,
    request_id: str = "init"
) -> EmbeddingsManager:
    """
    Initialize embeddings system - convenience function
    
    Args:
        reference_docs_path: Path to reference documents
        vector_store_path: Path to vector store
        force_rebuild: Force rebuild of vector store
        request_id: Request ID for logging
        
    Returns:
        Initialized EmbeddingsManager
    """
    logger.info(
        "=" * 80,
        extra={'request_id': request_id}
    )
    logger.info(
        "INITIALIZING EMBEDDINGS SYSTEM",
        extra={'request_id': request_id}
    )
    logger.info(
        "=" * 80,
        extra={'request_id': request_id}
    )
    
    try:
        # Create embeddings manager
        manager = EmbeddingsManager(
            reference_docs_path=reference_docs_path,
            vector_store_path=vector_store_path,
            request_id=request_id
        )
        
        # Check if vector store exists
        vector_store_file = os.path.join(vector_store_path, "faiss_index.faiss")
        
        if os.path.exists(vector_store_file) and not force_rebuild:
            logger.info(
                "Loading existing vector store...",
                extra={'request_id': request_id}
            )
            manager.load_vector_store()
        else:
            logger.info(
                "Building new vector store...",
                extra={'request_id': request_id}
            )
            
            # Load and chunk documents
            documents = manager.load_reference_documents()
            
            if not documents:
                logger.warning(
                    "No reference documents found. Vector store will be empty.",
                    extra={'request_id': request_id}
                )
            else:
                chunked_docs = manager.chunk_documents(documents)
                manager.create_vector_store(chunked_docs)
        
        # Get stats
        stats = manager.get_vector_store_stats()
        
        logger.info(
            "=" * 80,
            extra={'request_id': request_id}
        )
        logger.info(
            f"✓ EMBEDDINGS SYSTEM INITIALIZED",
            extra={'request_id': request_id}
        )
        logger.info(
            f"Status: {stats['status']}, Vectors: {stats.get('total_vectors', 0)}",
            extra={'request_id': request_id}
        )
        logger.info(
            "=" * 80,
            extra={'request_id': request_id}
        )
        
        return manager
        
    except Exception as e:
        logger.error(
            f"Failed to initialize embeddings system: {str(e)}",
            extra={'request_id': request_id},
            exc_info=True
        )
        raise


# CLI entry point for building vector store
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - [%(request_id)s] - %(message)s'
    )
    
    force_rebuild = "--rebuild" in sys.argv
    
    try:
        manager = initialize_embeddings(force_rebuild=force_rebuild)
        print("\n✓ Embeddings system ready!")
        print(f"Vector store stats: {manager.get_vector_store_stats()}")
        
    except Exception as e:
        print(f"\n✗ Failed: {str(e)}")
        sys.exit(1)