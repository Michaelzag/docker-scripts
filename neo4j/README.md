# Neo4j Graph Database

Graph database with APOC and Graph Data Science plugins.

## Quick Start

```bash
cp .env.example .env
docker compose up -d
```

## Access

- **Browser UI**: http://localhost:8474
- **Bolt**: `bolt://localhost:8487`
- **Default credentials**: neo4j / changeme

## Data

- Database files: Docker volume `neo4j-data`
- Import CSV/JSON: Place files in `./import` directory

## MCP Server

IDE configuration available in `mcp.json` for AI assistants.

## Ports

Change first 2 digits in `.env` to run multiple environments:
- Dev: 84xx
- Test: 74xx  
- Staging: 94xx