# Qdrant Vector Database

Vector database for similarity search and embeddings storage.

## Quick Start

```bash
cp .env.example .env
docker compose up -d
```

## Access

- **HTTP API**: http://localhost:8633
- **gRPC API**: `localhost:8634`
- **Dashboard**: http://localhost:8633/dashboard

## Data

- Vector data: Docker volume `qdrant-storage`
- Database files: Docker volume `qdrant-db`

## Ports

Change first 2 digits in `.env` to run multiple environments:
- Dev: 86xx
- Test: 76xx
- Staging: 96xx