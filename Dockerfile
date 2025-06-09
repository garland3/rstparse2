# ---- Base Stage ----
# Use an official Python slim image for a smaller footprint
FROM python:3.11-slim as base

# Set the working directory
WORKDIR /app

# Install system dependencies required for psycopg2
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# ---- Builder Stage ----
# This stage is for installing dependencies
FROM base as builder

# Install uv
RUN pip install uv

# Create a virtual environment using uv
RUN uv venv

# Copy only the dependency files to leverage Docker's layer caching
COPY pyproject.toml README.md ./

# Install dependencies into the virtual environment
RUN uv pip install --no-cache-dir .

# ---- Final Stage ----
# This is the final, lean image for our application
FROM base as final

# Copy the virtual environment from the builder stage
COPY --from=builder /app/.venv /app/.venv

# Add the virtual environment's bin directory to the PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy the rest of the application code
COPY . .

# Download static assets (requires requests which should be in dependencies)
RUN python download_assets.py

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "main.py"]
