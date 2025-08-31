# IntelliSupport RAG System 🤖

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-orange.svg)](https://openai.com/)

Enterprise-grade Retrieval-Augmented Generation (RAG) system for intelligent customer support automation. Built with modern AI technologies and production-ready architecture.

## 🚀 Features

- **🧠 Advanced RAG Pipeline**: Hybrid retrieval with semantic and keyword search
- **🌍 Multi-Language Support**: English, Hindi, Tamil with automatic detection
- **⚡ High Performance**: Sub-4 second response times with confidence scoring
- **📊 Production Ready**: Comprehensive error handling, logging, and monitoring
- **🔗 Citation Support**: Transparent source attribution for trustworthy responses
- **📱 RESTful API**: FastAPI with automatic documentation and validation
- **🧪 Fully Tested**: End-to-end testing with performance benchmarks

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   User Query    │───▶│  RAG Pipeline   │───▶│   AI Response   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐
                    │  Vector Store   │
                    │   (Embeddings)  │
                    └─────────────────┘
```

### Core Components

1. **Text Processing Engine** - Multi-language text cleaning and preprocessing
2. **Embedding Service** - OpenAI text-embedding-3-small integration
3. **Vector Retrieval** - Hybrid search with relevance scoring
4. **Response Generator** - GPT-4 powered contextual response generation
5. **API Layer** - FastAPI with comprehensive validation and error handling

## 🛠️ Quick Start

### Prerequisites

```bash
# Python 3.8+ required
python --version

# Install dependencies
pip install openai python-dotenv datasets pandas scikit-learn numpy fastapi uvicorn langdetect
```

### Environment Setup

Create a `.env` file with your OpenAI API key:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

### Run the System

1. **Generate Dataset**:
```bash
cd enterprise-rag-system
python3 scripts/data_ingestion/download_dataset.py
```

2. **Generate Embeddings**:
```bash
python3 scripts/data_ingestion/test_embeddings.py
```

3. **Test RAG Pipeline**:
```bash
python3 test_rag_pipeline.py
```

4. **Start API Server**:
```bash
python3 simple_api.py
```

### Test the API

```bash
# Health check
curl http://localhost:8000/health

# Process a query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "How do I reset my password?", "user_id": "test_user"}'
```

## 📊 Performance Results

Our testing shows excellent performance across key metrics:

| Metric | Result |
|--------|--------|
| **Average Response Time** | 2-4 seconds |
| **Retrieval Accuracy** | 0.6-0.8+ similarity scores |
| **Response Quality** | Contextually accurate and helpful |
| **Multi-language Support** | ✅ English, Hindi, Tamil ready |
| **Error Handling** | ✅ Comprehensive coverage |

### Sample Test Results

```
🔍 Query: "How do I reset my password?"
🤖 Response: To reset your password, please follow these steps:
1. Go to the login page
2. Click on 'Forgot Password'
3. Enter your email address
4. Check your email for reset instructions
5. Follow the link in the email to create a new password
⏱️ Time: 2920ms | 📊 Confidence: High
```

## 🏢 Enterprise Features

- **Scalable Architecture**: Designed for high-volume customer support
- **Multi-tenant Support**: User context and personalization
- **Security First**: Input validation and content filtering
- **Monitoring Ready**: Health checks and performance metrics
- **Documentation**: Comprehensive API docs with FastAPI

## 📁 Project Structure

```
enterprise-rag-system/
├── src/                          # Core application code
│   ├── core/                     # RAG components
│   │   ├── embeddings/           # Embedding services
│   │   └── rag/                  # Retrieval and generation
│   ├── api/                      # FastAPI routes and schemas
│   ├── utils/                    # Utility functions
│   └── config/                   # Configuration management
├── scripts/                      # Data processing scripts
│   └── data_ingestion/           # Dataset and embedding scripts
├── data/                         # Data storage
│   ├── raw/                      # Raw dataset files
│   ├── processed/                # Processed documents
│   └── embeddings/               # Generated embeddings
├── tests/                        # Test suites
├── docs/                         # Documentation
└── monitoring/                   # Monitoring and metrics
```

## 🔧 Configuration

The system supports extensive configuration through environment variables:

```bash
# OpenAI Configuration
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# RAG Configuration
MAX_CONTEXT_LENGTH=4000
RESPONSE_TEMPERATURE=0.1
INCLUDE_CITATIONS=true
```

## 🧪 Testing

Run the complete test suite:

```bash
# Unit tests
python -m pytest tests/

# Integration tests
python3 test_rag_pipeline.py

# API tests
python3 simple_api.py &
curl http://localhost:8000/docs
```

## 🚀 Deployment

### Docker Deployment
```bash
docker build -t intellisupport-rag .
docker run -p 8000:8000 --env-file .env intellisupport-rag
```

### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

## 📈 Monitoring

The system includes comprehensive monitoring:

- **Health Endpoints**: `/health`, `/health/detailed`
- **Performance Metrics**: Response times, confidence scores
- **Error Tracking**: Structured logging and error handling
- **Usage Analytics**: Query patterns and user feedback

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for GPT-4 and embedding models
- FastAPI for the excellent web framework
- The open-source community for various tools and libraries

---

**Built with ❤️ by the IntelliSupport Team**

For questions or support, please open an issue or contact our team.
