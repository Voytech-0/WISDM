

docker build -f infra/db_init/Dockerfile . -t db-init:latest

docker compose -f infra/docker-compose.yaml up -d mysql

docker compose -f infra/docker-compose.yaml run db-init