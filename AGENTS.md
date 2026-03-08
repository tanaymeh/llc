# Local Claude Code
- This project consists of a locally running coding agent that runs in a simple loop with agent instructions and tools and executes coding tasks.
- It may be run directly (`uv run main.py`) or via a docker container (`scripts/dev-docker.sh`). When using Docker container, you may specify a mount folder (this is usually the project you want to work on).
- We use `uv` as a package manager in this project and strictly adhere to best Python coding practices
- For the LLM calls, we use Openrouter platform and all the models available on it.
- Whenever there is a new implementation or feature in this project, make sure the README.md is updated (concisely, no slop) and the `scripts/dev-docker.sh` (for building properly).