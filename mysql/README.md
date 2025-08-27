# MySQL Database

Relational database with Adminer web UI for database management.

## Quick Start

```bash
cp .env.example .env
docker compose up -d
```

## Access

- **Adminer UI**: http://localhost:8880
- **MySQL**: `localhost:8306`
- **Default credentials**: user / changeme (root / changeme)

## Data

- Database files: Docker volume `mysql-data`
- Init scripts: Docker volume `mysql-init`
- Configuration: `./my.cnf`

## Ports

Change first 2 digits in `.env` to run multiple environments:
- Dev: 83xx/88xx
- Test: 73xx/78xx
- Staging: 93xx/98xx