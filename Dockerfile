# Stage 1: Build environment using uv
FROM ghcr.io/astral-sh/uv:python3.11-bookworm-slim AS builder
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first (leverage Docker cache)
COPY uv.lock pyproject.toml ./
RUN uv sync --frozen --no-install-project --no-dev

# Copy the rest of the application
COPY . /app
RUN uv sync --frozen --no-dev

# Stage 2: Final runtime image
FROM python:3.11-slim-bookworm

# Install runtime dependencies (git is required by unsloth)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy application and pre-built virtual environment from builder
COPY --from=builder /app /app

# Place virtual environment executable at the front of the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Expose Streamlit default port
EXPOSE 8501

# Run the Streamlit application
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
