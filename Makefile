.PHONY: help install test lint format train app clean

help: ## Show this help message
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-15s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## Install project dependencies (including dev tools)
	uv sync --frozen

update: ## Update project dependencies
	uv lock
	uv sync

test: ## Run unit tests
	uv run pytest tests/ -v

lint: ## Run linter
	uv run ruff check .

format: ## Format code
	uv run ruff format .
	uv run ruff check --fix .

train: ## Run the training script
	PYTHONPATH=. uv run python scripts/train.py

evaluate: ## Run the evaluation script
	PYTHONPATH=. uv run python scripts/eval_model.py

upload: ## Upload model to Hugging Face
	PYTHONPATH=. uv run python scripts/upload_hf.py

visualize_data: ## Run pre-training data visualizations
	PYTHONPATH=. uv run python scripts/visualize_data.py

visualize_results: ## Run post-training result visualizations
	PYTHONPATH=. uv run python scripts/visualize_results.py

app: ## Run the Streamlit app
	uv run streamlit run app.py

clean: ## Clean up cache directories and artifacts
	rm -rf .ruff_cache/
	rm -rf .pytest_cache/
	rm -rf __pycache__/
	find . -type d -name __pycache__ -exec rm -rf {} +

test-inference: ## Run CLI inference tests
	PYTHONPATH=. uv run python scripts/inference.py
