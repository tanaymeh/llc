ROOT_DIR := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
WORKSPACE ?= $(ROOT_DIR)
DOCKER_SCRIPT := $(ROOT_DIR)/scripts/dev-docker.sh

.PHONY: run

run:
	@if [ ! -d "$(WORKSPACE)" ]; then \
		echo "Workspace not found: $(WORKSPACE)"; \
		echo "Override with: make run WORKSPACE=/path/to/project"; \
		exit 1; \
	fi
	@"$(DOCKER_SCRIPT)" "$(WORKSPACE)"
