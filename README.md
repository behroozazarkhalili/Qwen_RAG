# Qwen3 RAG System

A production-ready Retrieval-Augmented Generation (RAG) system built with Qwen3 models, offering complete document processing, semantic retrieval, reranking, and answer generation capabilities.

## 🚀 Features

- **📄 PDF Document Processing**: Intelligent chunking and text extraction
- **🧠 Qwen3 Embeddings**: Semantic search using `Qwen/Qwen3-Embedding-0.6B`
- **🔍 Advanced Retrieval**: ChromaDB vector database with cosine similarity
- **📊 Smart Reranking**: Precision improvement with `Qwen/Qwen3-Reranker-0.6B`
- **💬 Answer Generation**: Natural language responses using `Qwen/Qwen2.5-1.5B-Instruct`
- **🏗️ Modular Architecture**: Clean, extensible design following RAG best practices
- **💾 Persistent Storage**: ChromaDB with automatic data persistence
- **⚡ Memory Management**: Automatic model cleanup and GPU memory optimization

## 🏗️ Architecture

The system implements a complete RAG pipeline:

1. **Document Processing** → Load and chunk PDF documents
2. **Embedding** → Convert text to vectors using Qwen3-Embedding
3. **Storage** → Store in ChromaDB vector database
4. **Retrieval** → Find relevant documents using semantic similarity
5. **Reranking** → Improve precision with Qwen3-Reranker
6. **Generation** → Generate natural language answers with Qwen3 LLM

## 📋 Requirements

- Python 3.8+
- CUDA-compatible GPU (recommended)
- 8GB+ GPU memory for optimal performance

## 🔧 Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Qwen_RAG
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Required packages:
- `torch>=2.0.0`
- `transformers>=4.51.0`
- `sentence-transformers>=2.7.0`
- `vllm>=0.9.0`
- `flash-attn>=2.0.0`
- `chromadb>=0.4.0`
- `PyPDF2>=3.0.0`
- `numpy>=1.21.0`

## 🚀 Quick Start

### Basic Usage

```python
from qwen3_rag import Qwen3RAG, RAGConfig

# Initialize RAG system
config = RAGConfig(
    device="cuda:0",
    collection_name="my_documents",
    persist_directory="./my_chroma_db"
)
rag = Qwen3RAG(config)

# Add PDF documents
pdf_paths = ["./data/document1.pdf", "./data/document2.pdf"]
rag.add_pdf_documents(pdf_paths)

# Query the system
question = "What is the main topic discussed in the documents?"
result = rag.answer(question, use_reranker=True)

print(f"Question: {result['question']}")
print(f"Answer: {result['answer']}")
print(f"Sources: {result['sources']}")
```

### Simple Chat Interface

```python
# Simple chat interface
answer = rag.chat("What are the key findings?", use_reranker=True)
print(answer)
```

### Retrieval Only

```python
# Retrieve relevant documents without answer generation
results = rag.query("machine learning", use_reranker=True)
print(f"Found {len(results['documents'])} relevant documents")
for doc, score in zip(results['documents'], results['rerank_scores']):
    print(f"Score: {score:.3f} - {doc[:100]}...")
```

## ⚙️ Configuration

The `RAGConfig` class provides comprehensive configuration options:

```python
config = RAGConfig(
    # Model configurations
    embedding_model="Qwen/Qwen3-Embedding-0.6B",
    reranker_model="Qwen/Qwen3-Reranker-0.6B", 
    generator_model="Qwen/Qwen2.5-1.5B-Instruct",
    
    # Document processing
    chunk_size=512,
    chunk_overlap=50,
    
    # Retrieval settings
    top_k_retrieval=20,
    top_k_rerank=5,
    similarity_threshold=0.0,
    
    # Generation settings
    max_context_length=4000,
    generation_temperature=0.1,
    generation_max_tokens=512,
    
    # System settings
    device="cuda:0",
    collection_name="qwen3_rag",
    persist_directory="./chroma_db"
)
```

## 📁 Project Structure

```
Qwen_RAG/
├── qwen3_rag.py              # Main RAG system implementation
├── Qwen3_RAG_Clean.ipynb     # Comprehensive example notebook
├── requirements.txt          # Python dependencies
├── data/                     # Sample documents
│   ├── Amazon-com-Inc-2023-Shareholder-Letter.pdf
│   └── Constitutional AI.pdf
└── my_chroma_db/            # ChromaDB storage (auto-created)
    └── chroma.sqlite3
```

## 💡 Key Components

### `Qwen3RAG` - Main RAG System
The primary interface for the RAG system with methods for:
- `add_pdf_documents()` - Add PDF documents to the knowledge base
- `query()` - Retrieve relevant documents
- `answer()` - Generate complete RAG responses
- `chat()` - Simple chat interface

### `DocumentProcessor`
Handles PDF loading and intelligent text chunking with configurable overlap.

### `Qwen3Embedder`
Wrapper for Qwen3 embedding model with optimized performance settings.

### `Qwen3Reranker`
Advanced reranking using Qwen3 reranker model for improved retrieval precision.

### `Qwen3Generator`
Text generation using Qwen3 instruction-tuned models with proper prompt templates.

### `VectorStore`
ChromaDB integration with persistent storage and similarity search.

## 🎯 Usage Examples

### Example 1: Corporate Document Analysis

```python
# Load corporate documents
rag = Qwen3RAG(config)
rag.add_pdf_documents(["./data/Amazon-com-Inc-2023-Shareholder-Letter.pdf"])

# Query financial information
result = rag.answer("What was Amazon's total revenue growth in 2023?")
print(result['answer'])
# Output: "Amazon's total revenue grew 12% year-over-year from $514B to $575B in 2023."
```

### Example 2: Research Paper Analysis

```python
# Configure for research documents
config = RAGConfig(
    chunk_size=768,  # Larger chunks for research papers
    top_k_rerank=3,
    generation_max_tokens=1024
)

rag = Qwen3RAG(config)
rag.add_pdf_documents(["./data/research_paper.pdf"])

result = rag.answer("What are the main contributions of this research?")
```

### Example 3: Multi-Document Comparison

```python
# Load multiple documents
rag.add_pdf_documents([
    "./data/document1.pdf",
    "./data/document2.pdf", 
    "./data/document3.pdf"
])

# Compare information across documents
result = rag.answer("Compare the approaches discussed in these documents")
```

## 🔧 Advanced Features

### Memory Management
The system automatically manages GPU memory by cleaning up unused models:

```python
# Automatic cleanup between reranker and generator
result = rag.answer(question, use_reranker=True)

# Manual cleanup
rag.cleanup()
```

### Custom Prompts
Extend the `RAGPromptTemplate` class for custom prompt engineering:

```python
class CustomPromptTemplate(RAGPromptTemplate):
    @staticmethod
    def get_rag_prompt(question: str, context: str) -> str:
        return f"Custom prompt: {question}\nContext: {context}"
```

### Retrieval Options
Choose between embedding-only or reranked retrieval:

```python
# Embedding-only retrieval (faster)
results = rag.query(question, use_reranker=False)

# Reranked retrieval (higher precision)
results = rag.query(question, use_reranker=True)
```

## 📊 Performance

- **Embedding Model**: Qwen3-Embedding-0.6B (~600M parameters)
- **Reranker Model**: Qwen3-Reranker-0.6B (~600M parameters)
- **Generator Model**: Qwen2.5-1.5B-Instruct (~1.5B parameters)
- **Memory Usage**: ~4-6GB GPU memory (with automatic cleanup)
- **Processing Speed**: ~1-3 seconds per query (depending on context length)

## 🐛 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - Reduce `chunk_size` or `top_k_retrieval`
   - Enable automatic cleanup: `use_reranker=True`

2. **Model Download Issues**
   - Ensure stable internet connection
   - Models are downloaded automatically on first use

3. **PDF Processing Errors**
   - Check PDF file permissions and format
   - Some complex PDFs may require preprocessing

### Performance Optimization

```python
# Optimize for speed
config = RAGConfig(
    top_k_retrieval=10,  # Reduce retrieval count
    top_k_rerank=3,      # Reduce reranking count
    generation_max_tokens=256  # Shorter responses
)

# Optimize for quality
config = RAGConfig(
    top_k_retrieval=30,
    top_k_rerank=10,
    similarity_threshold=0.1,  # Higher threshold
    generation_max_tokens=1024
)
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:

- Additional document format support (DOCX, TXT, etc.)
- Alternative embedding models
- Batch processing capabilities
- REST API interface
- Web UI interface

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- **Qwen Team** for the excellent Qwen3 model series
- **ChromaDB** for the vector database implementation
- **Hugging Face** for the transformers library
- **Sentence Transformers** for embedding utilities

## 📞 Support

For questions and support:
1. Check the example notebook: `Qwen3_RAG_Clean.ipynb`
2. Review the configuration options in `RAGConfig`
3. Examine the modular components in `qwen3_rag.py`

---

**Note**: This system requires a CUDA-compatible GPU for optimal performance. CPU-only execution is possible but significantly slower.