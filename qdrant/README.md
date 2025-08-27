# Qdrant Vector Database

## Docker Container

Vector database for similarity search and embeddings storage.

### Quick Start

```bash
cp .env.example .env
docker compose up -d
```

### Access

- **HTTP API**: http://localhost:6333 (default)
- **gRPC API**: `localhost:6334` (default)
- **Dashboard**: http://localhost:6333/dashboard (default)

### Data

- Vector data: Docker volume `qdrant-storage`
- Database files: Docker volume `qdrant-db`

---

## MCP Server

Official Qdrant Model Context Protocol (MCP) server providing semantic memory capabilities.

IDE configuration available in `mcp.json` for AI assistants.

**Package**: `mcp-server-qdrant` - Store and retrieve information using vector embeddings and semantic search.

### Features

- Store information in Qdrant database using vector embeddings
- Retrieve relevant information using semantic search
- Semantic memory layer on top of Qdrant vector database
- Configurable embedding models for vector generation

### Configuration Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `QDRANT_URL` | Qdrant server URL | `http://localhost:6333` |
| `QDRANT_API_KEY` | Authentication for Qdrant | *(none)* |
| `COLLECTION_NAME` | Name of the vector collection | `mcp_memory` |
| `EMBEDDING_MODEL` | Embedding model for vectors | `sentence-transformers/all-MiniLM-L6-v2` |

### Available Tools

- `qdrant-store`: Store information in Qdrant database
- `qdrant-find`: Retrieve relevant information using semantic search

### Requirements

- Python 3.8+
- Qdrant server
- uvx (for installation and usage)