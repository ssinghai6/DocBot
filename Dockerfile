FROM python:3.12-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m appuser

# Copy requirements first for layer caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/ api/

# Switch to non-root user
USER appuser

# Start server using Railway's PORT env var
EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.index:app --host 0.0.0.0 --port ${PORT:-8000}"]
