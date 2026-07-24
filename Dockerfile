# Dockerfile for the Moon Classifier API (Tutorial #2's app)
FROM python:3.11-slim

WORKDIR /app

# --- 1. Dependencies FIRST: these layers are CACHED until the files
# --- named here change, so code edits don't re-download PyTorch.
COPY requirements-docker.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements-docker.txt

# --- 2. Code and model AFTER: these change often, and only these
# --- layers rebuild when they do.
COPY src/ src/
COPY app/ app/
COPY models/ models/

# --- 3. Documentation of the port the app listens on inside the box.
EXPOSE 8000

# --- 4. What runs when the container starts.
# --host 0.0.0.0 is REQUIRED in a container: 127.0.0.1 inside the box
# means "this box only" and would be unreachable even from your own PC.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]