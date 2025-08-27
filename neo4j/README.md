# Neo4j Graph Database

## Docker Container

Graph database with APOC and Graph Data Science plugins.

### Quick Start

```bash
cp .env.example .env
docker compose up -d
```

### Access

- **Browser UI**: http://localhost:7474 (default)
- **Bolt**: `bolt://localhost:7687` (default)
- **Default credentials**: neo4j / changeme

### Data

- Database files: Docker volume `neo4j-data`
- Import CSV/JSON: Place files in `./import` directory

---

## MCP Server

IDE configuration available in `mcp.json` for AI assistants.

**Package**: `mcp-neo4j-cypher` - Execute Cypher queries and browse graph schema via MCP tools.