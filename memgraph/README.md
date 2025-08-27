# Memgraph Graph Database

Graph database with Lab web UI for visualization and queries.

## Quick Start

```bash
cp .env.example .env
docker compose up -d
```

## Access

- **Lab UI**: http://localhost:8550
- **Bolt**: `bolt://localhost:8587`
- **Log API**: http://localhost:8544
- **Default credentials**: memgraph / changeme

## Data

- Database files: Docker volume `memgraph-db-data`
- Configuration: Docker volume `memgraph-db-conf`

## Ports

Change first 2 digits in `.env` to run multiple environments:
- Dev: 85xx
- Test: 75xx
- Staging: 95xx