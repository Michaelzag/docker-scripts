# MySQL Database

## Docker Container

Relational database with Adminer web UI for database management.

### Quick Start

```bash
cp .env.example .env
docker compose up -d
```

### Access

- **Adminer UI**: http://localhost:8080 (default)
- **MySQL**: `localhost:3306` (default)
- **Default credentials**: user / changeme (root / changeme)

### Data

- Database files: Docker volume `mysql-data`
- Init scripts: Docker volume `mysql-init`
- Configuration: `./my.cnf`

---

## MCP Server mysql-mcp-zag

### Configuration Arguments

#### Required Arguments
| Argument | Description | Default |
|----------|-------------|---------|
| `--user` | MySQL username | **(required)** |
| `--password` | MySQL password | **(required)** |
| `--database` | MySQL database name | **(required)** |

#### Database Connection (Optional)
| Argument | Description | Default |
|----------|-------------|---------|
| `--host` | MySQL server host | `localhost` |
| `--port` | MySQL server port | `3306` |

#### SSL Configuration (Optional)
| Argument | Description | Default |
|----------|-------------|---------|
| `--ssl-ca` | Path to SSL CA certificate file | *(none - SSL auto-negotiated)* |
| `--ssl-cert` | Path to SSL client certificate file | *(none)* |
| `--ssl-key` | Path to SSL client private key file | *(none)* |
| `--ssl-disabled` | Disable SSL connection entirely | `false` |

**Note:** If `--ssl-cert` is provided, `--ssl-key` must also be provided, and vice versa.

#### Advanced Options (Optional)
| Argument | Description | Default |
|----------|-------------|---------|
| `--charset` | Character set for the connection | `utf8mb4` |
| `--collation` | Collation for the connection | `utf8mb4_unicode_ci` |
| `--sql-mode` | MySQL SQL mode | `TRADITIONAL` |

### Available Tools

- `execute_sql`: Execute SQL queries

### Available Resources

- `mysql://tables`: List all tables
- `mysql://tables/{table}`: Describe table structure

### Requirements

- Python 3.13+
- uvx (for installation and usage)