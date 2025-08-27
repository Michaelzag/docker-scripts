# Memgraph Graph Database

## Docker Container

Graph database with Lab web UI for visualization and queries.

### Quick Start

```bash
cp .env.example .env
docker compose up -d
```

### Access

- **Lab UI**: http://localhost:3000 (default)
- **Bolt**: `bolt://localhost:7687` (default)
- **Log API**: http://localhost:7444 (default)
- **Default credentials**: memgraph / changeme

### Data

- Database files: Docker volume `memgraph-db-data`
- Configuration: Docker volume `memgraph-db-conf`

---

## MCP Server

Compatible with Neo4j MCP server due to shared Bolt protocol and Cypher query language.

IDE configuration available in `mcp.json` for AI assistants.

**Package**: `mcp-neo4j-cypher` - Execute Cypher queries and browse graph schema via MCP tools (compatible with Memgraph).

### Features

- Execute Cypher queries on Memgraph database
- Browse database schema and structure
- Full Bolt protocol compatibility with Neo4j drivers
- Memgraph v2.11+ has default compatibility (no configuration needed)

### Requirements

- Python 3.8+
- Memgraph server (v2.11+ recommended for best compatibility)
- uvx (for installation and usage)

### Compatibility Notes

- Uses Neo4j MCP server (`mcp-neo4j-cypher`) with Memgraph backend
- Memgraph implements Bolt v4/v4.1 protocol for Neo4j driver compatibility
- Same connection format as Neo4j (bolt://localhost:7687)