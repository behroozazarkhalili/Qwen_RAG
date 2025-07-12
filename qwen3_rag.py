"""
Qwen3 RAG System with PDF Support
A production-ready RAG implementation using Qwen3 embeddings and reranker.
"""

import os
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import math

import torch
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import PyPDF2
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig
import gc

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class RAGConfig:
    """Configuration for Qwen3 RAG system"""
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B"
    reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    generator_model: str = "Qwen/Qwen2.5-1.5B-Instruct"  # LLM for answer generation
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k_retrieval: int = 20
    top_k_rerank: int = 5
    similarity_threshold: float = 0.0
    device: str = "cuda:0"
    collection_name: str = "qwen3_rag"
    persist_directory: str = "./chroma_db"
    max_context_length: int = 4000  # Max context for LLM
    generation_temperature: float = 0.1
    generation_max_tokens: int = 512


@dataclass
class Document:
    """Document representation"""
    content: str
    metadata: Dict[str, Any]
    
    
class DocumentProcessor:
    """Handles document loading and chunking"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        
    def load_pdf(self, pdf_path: str) -> List[Document]:
        """Load and extract text from PDF"""
        try:
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                text = ""
                for page_num, page in enumerate(reader.pages):
                    text += f"\n--- Page {page_num + 1} ---\n"
                    text += page.extract_text()
                    
            return [Document(
                content=text,
                metadata={
                    "source": pdf_path,
                    "type": "pdf",
                    "pages": len(reader.pages)
                }
            )]
        except Exception as e:
            logger.error(f"Error loading PDF {pdf_path}: {e}")
            raise
            
    def chunk_documents(self, documents: List[Document]) -> List[Document]:
        """Split documents into chunks"""
        chunks = []
        
        for doc in documents:
            text = doc.content
            words = text.split()
            
            for i in range(0, len(words), self.config.chunk_size - self.config.chunk_overlap):
                chunk_words = words[i:i + self.config.chunk_size]
                chunk_text = " ".join(chunk_words)
                
                chunk_metadata = doc.metadata.copy()
                chunk_metadata.update({
                    "chunk_id": len(chunks),
                    "start_word": i,
                    "end_word": min(i + self.config.chunk_size, len(words))
                })
                
                chunks.append(Document(
                    content=chunk_text,
                    metadata=chunk_metadata
                ))
                
        logger.info(f"Created {len(chunks)} chunks from {len(documents)} documents")
        return chunks


class Qwen3Embedder:
    """Qwen3 embedding model wrapper"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._setup_environment()
        self._load_model()
        
    def _setup_environment(self):
        """Setup CUDA environment"""
        os.environ["CUDA_VISIBLE_DEVICES"] = self.config.device.split(":")[-1]
        
    def _load_model(self):
        """Load the embedding model"""
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        self.model = SentenceTransformer(
            self.config.embedding_model,
            model_kwargs={
                "attn_implementation": "flash_attention_2",
                "device_map": "auto",
                "torch_dtype": "float16"
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
        
    def embed_texts(self, texts: List[str], prompt_name: Optional[str] = None) -> np.ndarray:
        """Generate embeddings for texts"""
        try:
            if prompt_name:
                embeddings = self.model.encode(texts, prompt_name=prompt_name)
            else:
                embeddings = self.model.encode(texts)
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
            
    def embed_query(self, query: str) -> np.ndarray:
        """Generate embedding for query with query prompt"""
        return self.embed_texts([query], prompt_name="query")[0]


class Qwen3Reranker:
    """Qwen3 reranker model wrapper using transformers"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load the reranker model"""
        logger.info(f"Loading reranker model: {self.config.reranker_model}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.reranker_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.reranker_model,
                torch_dtype=torch.float16,
                device_map={"": self.config.device},
                trust_remote_code=True
            )
            self.model.eval()
        except Exception as e:
            logger.error(f"Failed to load reranker model: {e}")
            raise
        
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        self.max_length = 2048
        self.true_token = self.tokenizer("yes", add_special_tokens=False).input_ids[0]
        self.false_token = self.tokenizer("no", add_special_tokens=False).input_ids[0]
        
    def _format_instruction(self, instruction: str, query: str, doc: str) -> List[Dict[str, str]]:
        """Format instruction for reranker"""
        return [
            {"role": "system", "content": "Judge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\"."},
            {"role": "user", "content": f"<Instruct>: {instruction}\n\n<Query>: {query}\n\n<Document>: {doc}"}
        ]
        
    def rerank(self, query: str, documents: List[str], instruction: str = None) -> List[float]:
        """Rerank documents based on query using transformers"""
        if instruction is None:
            instruction = 'Given a web search query, retrieve relevant passages that answer the query'
            
        try:
            scores = []
            
            for doc in documents:
                # Format the prompt
                messages = self._format_instruction(instruction, query, doc)
                
                # Apply chat template
                prompt = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                prompt += self.suffix
                
                # Tokenize
                inputs = self.tokenizer(
                    prompt, 
                    return_tensors="pt", 
                    max_length=self.max_length, 
                    truncation=True,
                    padding=True
                ).to(self.model.device)
                
                # Generate with logits
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0, -1, :]  # Last token logits
                    
                    # Get probabilities for yes/no tokens
                    true_logit = logits[self.true_token].item()
                    false_logit = logits[self.false_token].item()
                    
                    # Convert to probabilities
                    true_score = math.exp(true_logit)
                    false_score = math.exp(false_logit)
                    score = true_score / (true_score + false_score)
                    scores.append(score)
                    
            return scores
            
        except Exception as e:
            logger.error(f"Error in reranking: {e}")
            raise
            
    def cleanup(self):
        """Cleanup reranker resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Reranker cleaned up successfully")
        except Exception as e:
            logger.warning(f"Reranker cleanup warning: {e}")


class VectorStore:
    """ChromaDB vector store wrapper"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self._setup_chroma()
        
    def _setup_chroma(self):
        """Setup ChromaDB client and collection"""
        self.client = chromadb.PersistentClient(
            path=self.config.persist_directory,
            settings=Settings(anonymized_telemetry=False)
        )
        
        try:
            self.collection = self.client.get_collection(self.config.collection_name)
            logger.info(f"Loaded existing collection: {self.config.collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {self.config.collection_name}")
            
    def add_documents(self, documents: List[Document], embeddings: np.ndarray):
        """Add documents with embeddings to vector store"""
        ids = [f"doc_{i}" for i in range(len(documents))]
        texts = [doc.content for doc in documents]
        metadatas = [doc.metadata for doc in documents]
        
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings.tolist(),
            metadatas=metadatas
        )
        logger.info(f"Added {len(documents)} documents to vector store")
        
    def search(self, query_embedding: np.ndarray, k: int) -> Tuple[List[str], List[Dict], List[float]]:
        """Search for similar documents"""
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=k
        )
        
        documents = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0]
        
        # Convert distances to similarities (ChromaDB returns cosine distances)
        similarities = [1 - d for d in distances]
        
        return documents, metadatas, similarities
        
    def count(self) -> int:
        """Get document count"""
        return self.collection.count()


class RAGPromptTemplate:
    """RAG prompt templates"""
    
    @staticmethod
    def get_rag_prompt(question: str, context: str) -> str:
        """Generate RAG prompt with context and question"""
        return f"""Based on the following context, please answer the question. When possible, cite the specific sources using [Source: filename] format. If the answer cannot be found in the context, say "I cannot find the answer in the provided context."

Context:
{context}

Question: {question}

Answer:"""

    @staticmethod
    def get_system_prompt() -> str:
        """Get system prompt for RAG"""
        return """You are a helpful AI assistant that answers questions based on the provided context. Always base your answers on the given context and cite relevant sources using [Source: filename] format when referencing specific information. Be precise and include citations in your response. If you cannot find the answer in the context, be honest about it."""


class Qwen3Generator:
    """Qwen3 text generation model wrapper using transformers"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._load_model()
        
    def _load_model(self):
        """Load the generator model"""
        logger.info(f"Loading generator model: {self.config.generator_model}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.generator_model)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.config.generator_model,
                torch_dtype=torch.float16,
                device_map={"": self.config.device},
                trust_remote_code=True
            )
            self.model.eval()
            
            # Set pad token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # Generation config
            self.generation_config = GenerationConfig(
                temperature=self.config.generation_temperature,
                max_new_tokens=self.config.generation_max_tokens,
                top_p=0.9,
                repetition_penalty=1.1,
                do_sample=True if self.config.generation_temperature > 0 else False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
            
        except Exception as e:
            logger.error(f"Failed to load generator model: {e}")
            raise
        
    def generate_answer(self, question: str, context: str) -> str:
        """Generate answer based on question and context"""
        try:
            # Prepare messages
            system_prompt = RAGPromptTemplate.get_system_prompt()
            user_prompt = RAGPromptTemplate.get_rag_prompt(question, context)
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # Apply chat template
            prompt = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Tokenize input
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                max_length=2048,
                truncation=True
            ).to(self.model.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    generation_config=self.generation_config
                )
                
            # Decode response (skip input tokens)
            response_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            answer = self.tokenizer.decode(response_tokens, skip_special_tokens=True).strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
            
    def cleanup(self):
        """Cleanup generator resources"""
        try:
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            gc.collect()
            torch.cuda.empty_cache()
            logger.info("Generator cleaned up successfully")
        except Exception as e:
            logger.warning(f"Generator cleanup warning: {e}")


class Qwen3RAG:
    """Main RAG system class"""
    
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.processor = DocumentProcessor(self.config)
        self.embedder = Qwen3Embedder(self.config)
        self.reranker = None  # Lazy loading
        self.generator = None  # Lazy loading
        self.vector_store = VectorStore(self.config)
        
    def add_pdf_documents(self, pdf_paths: List[str]):
        """Add PDF documents to the RAG system"""
        all_documents = []
        
        for pdf_path in pdf_paths:
            logger.info(f"Processing PDF: {pdf_path}")
            documents = self.processor.load_pdf(pdf_path)
            chunked_docs = self.processor.chunk_documents(documents)
            all_documents.extend(chunked_docs)
            
        if all_documents:
            logger.info("Generating embeddings for documents...")
            texts = [doc.content for doc in all_documents]
            embeddings = self.embedder.embed_texts(texts)
            
            self.vector_store.add_documents(all_documents, embeddings)
            logger.info(f"Successfully added {len(all_documents)} document chunks")
            
    def query(self, question: str, use_reranker: bool = True) -> Dict[str, Any]:
        """Query the RAG system"""
        logger.info(f"Processing query: {question}")
        
        # Generate query embedding
        query_embedding = self.embedder.embed_query(question)
        
        # Retrieve candidates
        k = self.config.top_k_rerank if use_reranker else self.config.top_k_retrieval
        documents, metadatas, similarities = self.vector_store.search(query_embedding, k)
        
        # Filter by similarity threshold
        filtered_results = [
            (doc, meta, sim) for doc, meta, sim in zip(documents, metadatas, similarities)
            if sim >= self.config.similarity_threshold
        ]
        
        if not filtered_results:
            return {
                "query": question,
                "documents": [],
                "similarities": [],
                "rerank_scores": [],
                "metadatas": []
            }
            
        documents, metadatas, similarities = zip(*filtered_results)
        documents, metadatas, similarities = list(documents), list(metadatas), list(similarities)
        
        rerank_scores = []
        if use_reranker and len(documents) > 1:
            if self.reranker is None:
                self.reranker = Qwen3Reranker(self.config)
                
            logger.info("Reranking documents...")
            rerank_scores = self.reranker.rerank(question, documents)
            
            # Sort by rerank scores
            sorted_indices = sorted(range(len(rerank_scores)), key=lambda i: rerank_scores[i], reverse=True)
            documents = [documents[i] for i in sorted_indices]
            metadatas = [metadatas[i] for i in sorted_indices]
            similarities = [similarities[i] for i in sorted_indices]
            rerank_scores = [rerank_scores[i] for i in sorted_indices]
            
            # Take top k after reranking
            documents = documents[:self.config.top_k_rerank]
            metadatas = metadatas[:self.config.top_k_rerank]
            similarities = similarities[:self.config.top_k_rerank]
            rerank_scores = rerank_scores[:self.config.top_k_rerank]
            
        return {
            "query": question,
            "documents": documents,
            "similarities": similarities,
            "rerank_scores": rerank_scores,
            "metadatas": metadatas
        }
        
    def get_context(self, question: str, use_reranker: bool = True) -> str:
        """Get formatted context for the question"""
        results = self.query(question, use_reranker)
        
        if not results["documents"]:
            return "No relevant documents found."
            
        context_parts = []
        total_length = 0
        
        for i, (doc, metadata) in enumerate(zip(results["documents"], results["metadatas"])):
            source = metadata.get("source", "Unknown")
            chunk_id = metadata.get("chunk_id", i)
            
            # Format context entry
            context_entry = f"[Source: {source}, Chunk: {chunk_id}]\n{doc}"
            
            # Check if adding this would exceed max context length
            if total_length + len(context_entry) > self.config.max_context_length:
                logger.info(f"Context truncated at {i} documents due to length limit")
                break
                
            context_parts.append(context_entry)
            total_length += len(context_entry)
            
        return "\n\n".join(context_parts)
        
    def answer(self, question: str, use_reranker: bool = True) -> Dict[str, Any]:
        """Generate complete RAG answer for the question"""
        logger.info(f"Generating RAG answer for: {question}")
        
        # Get relevant context
        context = self.get_context(question, use_reranker)
        
        if context == "No relevant documents found.":
            return {
                "question": question,
                "answer": "I cannot find relevant information to answer your question.",
                "context": context,
                "sources": []
            }
        
        # Clean up reranker before loading generator to save memory
        if self.reranker and use_reranker:
            logger.info("Cleaning up reranker to free memory for generator...")
            self.reranker.cleanup()
            self.reranker = None
            
        # Initialize generator if needed
        if self.generator is None:
            logger.info("Loading generator model...")
            self.generator = Qwen3Generator(self.config)
            
        # Generate answer
        answer = self.generator.generate_answer(question, context)
        
        # Get source information
        retrieval_results = self.query(question, use_reranker=False)  # Skip reranker since we cleaned it
        sources = []
        citations = []
        
        for i, metadata in enumerate(retrieval_results["metadatas"]):
            source_name = metadata.get("source", "Unknown")
            chunk_id = metadata.get("chunk_id", "Unknown")
            page_info = metadata.get("pages", None)
            
            # Create detailed source info
            source_info = {
                "source": source_name,
                "chunk_id": chunk_id,
                "page": page_info
            }
            
            # Create citation text
            if source_name.endswith('.pdf'):
                source_display = source_name.split('/')[-1]  # Get filename only
                if page_info:
                    citation = f"[Source: {source_display}, Page {page_info}]"
                else:
                    citation = f"[Source: {source_display}]"
            else:
                citation = f"[Source: {source_name}]"
                
            if source_info not in sources:
                sources.append(source_info)
                citations.append(citation)
                
        return {
            "question": question,
            "answer": answer,
            "context": context,
            "sources": sources,
            "citations": citations,
            "retrieval_results": retrieval_results
        }
        
    def chat(self, question: str, use_reranker: bool = True) -> str:
        """Simple chat interface that returns just the answer"""
        result = self.answer(question, use_reranker)
        return result["answer"]
        
    def cleanup(self):
        """Cleanup resources"""
        if self.reranker:
            self.reranker.cleanup()
        if self.generator:
            self.generator.cleanup()
            
    def get_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        return {
            "document_count": self.vector_store.count(),
            "config": self.config
        }