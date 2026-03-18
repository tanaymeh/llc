ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
WORKSPACE ?= $(ROOT_DIR)
DOCKER_SCRIPT := $(ROOT_DIR)/scripts/dev-docker.sh
COMPOSE_FILE := $(ROOT_DIR)/docker-compose.yml

.PHONY: run down logs run-backend

run:
	@if [ ! -d "$(WORKSPACE)" ]; then \
		echo "Workspace not found: $(WORKSPACE)"; \
		echo "Override with: make run WORKSPACE=/path/to/project"; \
		exit 1; \
	fi
	@WORKSPACE="$(WORKSPACE)" docker compose -f "$(COMPOSE_FILE)" up --build

down:
	@WORKSPACE="$(WORKSPACE)" docker compose -f "$(COMPOSE_FILE)" down

logs:
	@WORKSPACE="$(WORKSPACE)" docker compose -f "$(COMPOSE_FILE)" logs -f

run-backend:
	@"$(DOCKER_SCRIPT)" "$(WORKSPACE)"
