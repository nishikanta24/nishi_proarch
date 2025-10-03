# AI Cost & Insights Copilot

[![CI](https://github.com/yourusername/proarch/workflows/CI%20Pipeline/badge.svg)](https://github.com/yourusername/proarch/actions)
[![Python Version](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive AI-powered analytics platform for cloud cost management and optimization recommendations.

## 🚀 Features

- **Data Ingestion**: Automated pipeline for cloud billing data
- **KPI Analytics**: Real-time cost metrics and trend analysis
- **RAG Q&A**: Natural language queries about cloud costs with AI-powered answers
- **Cost Optimization**: Automated recommendations for cost savings
- **Token Usage Tracking**: Real-time monitoring of LLM costs and usage
- **Intelligent Caching**: Performance optimization for repeated queries

## 🛠️ Quick Start

### Prerequisites
- Python 3.8+
- SQLite (default) or PostgreSQL

### Installation

```bash
# Clone repository
git clone <repository-url>
cd proarch

# Install dependencies
make install

# For development
make dev-install
```

### Running the Application

```bash
# Start API server
make run-api

# Start frontend (in another terminal)
make run-frontend

# Ingest sample data
make run-ingestion
```

## 🧪 Testing & Quality

### Run Tests
```bash
# Run all tests
make test

# Run unit tests only
make test-unit

# Run with coverage
make coverage
```

### Code Quality
```bash
# Lint code
make lint

# Format code
make format

# Type checking
make type-check
```

## 🔄 CI/CD

This project uses GitHub Actions for continuous integration:

### Automated Checks
- **Multi-Python Testing**: Python 3.8-3.12 compatibility
- **Code Linting**: flake8 for code quality
- **Formatting**: black for consistent code style
- **Type Checking**: mypy for static type analysis
- **Unit Tests**: Comprehensive test suite
- **Docker Build**: Container build verification

### Local CI Simulation
```bash
# Run all CI checks locally
make ci-local
```

## 🏗️ Architecture

```
src/
├── api/           # FastAPI application and routes
├── ai/            # RAG pipeline and LLM integration
├── ingestion/     # Data pipeline and transformations
├── recommendations/# Cost optimization logic
├── transformations/# KPI calculations and analytics
└── ui/           # User interface components
```

## 📊 API Endpoints

- `GET /api/v1/kpi` - Key performance indicators
- `POST /api/v1/ask` - Natural language Q&A
- `POST /api/v1/recommendations` - Cost optimization suggestions
- `GET /api/v1/health` - Health check

## 🔧 Configuration

Environment variables:

```bash
# Database
DATABASE_URL=sqlite:///./data/cost_analytics.db

# AI/ML
MODEL_NAME=deepseek/deepseek-r1-0528-qwen3-8b:free
OPENROUTER_API_KEY=your-api-key

# Caching (New!)
ENABLE_CACHE=true
CACHE_TTL_MINUTES=30

# API
API_HOST=0.0.0.0
API_PORT=8000
```

## 🐳 Docker Deployment

### Quick Start with Docker Compose

```bash
# 1. Ensure you have .env file with your API keys
cp .env.example .env  # Edit with your keys

# 2. Full deployment (build + init data + start services)
./deploy.sh deploy

# 3. Check status
./deploy.sh status

# 4. View logs
./deploy.sh logs

# Access your application:
# - Frontend: http://localhost:8501
# - API: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

### Development Mode

```bash
# Start with hot reloading and volume mounting
./deploy.sh start-dev

# Or with make
make docker-compose-dev
```

### Production Mode

```bash
# Start with production optimizations
./deploy.sh start-prod

# Includes Nginx reverse proxy on port 80
```

### Manual Docker Commands

#### Single Container
```bash
make docker-build
make docker-run
```

#### Multi-Service with Docker Compose
```bash
# Start services
make docker-compose-up

# Start in background
make docker-compose-up-d

# Initialize data
make docker-compose-init

# View logs
make docker-compose-logs

# Stop services
make docker-compose-down
```

### Deployment Script Commands

The `deploy.sh` script provides convenient commands:

```bash
./deploy.sh check         # Verify Docker and environment setup
./deploy.sh build         # Build Docker images
./deploy.sh init          # Initialize database with sample data
./deploy.sh start         # Start services
./deploy.sh start-dev     # Start in development mode
./deploy.sh start-prod    # Start in production mode
./deploy.sh stop          # Stop services
./deploy.sh restart       # Restart services
./deploy.sh status        # Show service health
./deploy.sh logs          # Show all logs
./deploy.sh logs-api      # Show API logs only
./deploy.sh cleanup       # Clean up containers and images
```

### Docker Architecture

```
proarch-network
├── proarch-api (FastAPI)
│   ├── Port: 8000
│   ├── Health: /health
│   └── Volumes: ./data
├── proarch-frontend (Streamlit)
│   ├── Port: 8501
│   └── Depends on: proarch-api
└── proarch-init (Data initialization)
    └── Profiles: init (runs once)
```

### Environment Configuration

Required environment variables in `.env`:

```bash
# API Keys
OPENROUTER_API_KEY=your-openrouter-api-key

# Database
DATABASE_URL=sqlite:///./data/cost_analytics.db

# AI/ML
MODEL_NAME=deepseek/deepseek-r1-0528-qwen3-8b:free
VECTOR_STORE_PATH=./data/vector_store

# Caching
ENABLE_CACHE=true
CACHE_TTL_MINUTES=30

# API
API_HOST=0.0.0.0
API_PORT=8000
CLIENT_ORIGINS=http://localhost:3000,http://localhost:8501
```

### Production Considerations

- **Security**: Non-root user, minimal attack surface
- **Performance**: Multi-stage builds, optimized images
- **Monitoring**: Health checks, structured logging
- **Scalability**: Ready for container orchestration (Kubernetes)
- **Persistence**: Data volumes for database and vector store

## 📈 Observability

The application provides comprehensive observability:

- **Structured Logging**: Request IDs and detailed metrics
- **Token Usage Tracking**: Real-time LLM cost monitoring
- **Cache Metrics**: Hit rates and performance statistics
- **Request Latency**: Millisecond-precision timing
- **Health Checks**: System status monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Ensure CI passes: `make ci-local`
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
