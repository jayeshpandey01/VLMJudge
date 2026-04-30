# Name: Jayesh Pandey
# Summary: Docker configuration for containerizing the application.

FROM python:3.10-slim

WORKDIR /app

# Needed for git-based deps (requirements.txt uses git+https).
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*

COPY . /app

RUN pip install --no-cache-dir -r requirements_api.txt

# Best-effort install project deps (may include GPU-specific packages in real deployments).
RUN pip install --no-cache-dir -r requirements.txt || true

EXPOSE 8000

CMD ["python", "run_api.py", "--host", "0.0.0.0", "--port", "8000", "--config", "config.yaml"]

