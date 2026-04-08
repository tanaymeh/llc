ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
WORKSPACE ?= $(ROOT_DIR)
DOCKER_SCRIPT := $(ROOT_DIR)/scripts/dev-docker.sh
COMPOSE_FILE := $(ROOT_DIR)/docker-compose.yml
LANGFUSE_COMPOSE_FILE := $(ROOT_DIR)/docker-compose.langfuse.yml

.PHONY: run down logs run-backend langfuse-up langfuse-down langfuse-logs

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

langfuse-up:
	@set -a; \
	if [ -f "$(ROOT_DIR)/.env" ]; then . "$(ROOT_DIR)/.env"; fi; \
	if [ -z "$${LANGFUSE_INIT_PROJECT_PUBLIC_KEY:-}" ]; then \
		LANGFUSE_INIT_PROJECT_PUBLIC_KEY="$${LANGFUSE_PUBLIC_KEY:-pk-lf-llc-local-public-key}"; \
	fi; \
	if [ -z "$${LANGFUSE_INIT_PROJECT_SECRET_KEY:-}" ]; then \
		LANGFUSE_INIT_PROJECT_SECRET_KEY="$${LANGFUSE_SECRET_KEY:-sk-lf-llc-local-secret-key}"; \
	fi; \
	export LANGFUSE_INIT_PROJECT_PUBLIC_KEY LANGFUSE_INIT_PROJECT_SECRET_KEY; \
	docker compose -f "$(LANGFUSE_COMPOSE_FILE)" up -d

langfuse-down:
	@docker compose -f "$(LANGFUSE_COMPOSE_FILE)" down

langfuse-logs:
	@docker compose -f "$(LANGFUSE_COMPOSE_FILE)" logs -f
