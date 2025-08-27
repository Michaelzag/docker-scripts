# Grafana + Loki Monitoring

## Docker Container

Log aggregation with Loki and visualization with Grafana. Automatically collects Docker container logs.

### Quick Start

```bash
cp .env.example .env
docker compose up -d
```

### Access

- **Grafana**: http://localhost:7000 (default, anonymous access enabled)
- **Loki API**: http://localhost:7100 (default)

### Features

- Collects all Docker container logs automatically
- Extracts labels from Docker containers
- Pre-configured Loki datasource in Grafana
- Dashboard provisioning from `./dashboards` directory