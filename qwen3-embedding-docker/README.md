# ====Still Testing This===


# Codebase Indexing with Docker Model Runner

Setup Qwen3-Embedding-4B-GGUF for codebase indexing using Docker Model Runner.

## 🚀 Setup

### 1. Enable Docker Model Runner

**Docker Desktop:**
- Settings → AI tab → Enable "Enable Docker Model Runner"

**Docker Engine:**
```bash
sudo apt-get update
sudo apt-get install docker-model-plugin
docker model version
```

### 2. Start Service
```bash
cd qwen3-embedding-docker
docker compose up -d
```

## 🔧 Codebase Indexing Configuration

### Environment Variables (.env)
```bash
DOCKER_EMBEDDING_PORT=7801
DOCKER_MODEL=Qwen/Qwen3-Embedding-4B-GGUF
```

### IDE Integration
```
Base URL: http://localhost:7801/engines/llama.cpp/v1
API Key: dummy
Model: Qwen/Qwen3-Embedding-4B-GGUF
Dimensions: 2560
```

## 📡 API Usage

### Create Embeddings
```bash
curl -X POST "http://localhost:7801/engines/llama.cpp/v1/embeddings" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-Embedding-4B-GGUF",
    "input": ["function calculateSum(a, b) { return a + b; }"],
    "encoding_format": "float"
  }'
```
