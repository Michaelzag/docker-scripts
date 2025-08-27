# Grafana + Loki Monitoring

Log aggregation with Loki and visualization with Grafana. Automatically collects Docker container logs.

## Quick Start

```bash
cp .env.example .env
docker compose up -d
```

## Access

- **Grafana**: http://localhost:9700 (anonymous access enabled)
- **Loki API**: http://localhost:9710

## Features

- Collects all Docker container logs automatically
- Extracts labels from Docker containers
- Pre-configured Loki datasource in Grafana
- Dashboard provisioning from `./dashboards` directory

## Ports

Change first 2 digits in `.env` to run multiple environments:
- Dev: 97xx
- Test: 87xx
- Staging: 98xx