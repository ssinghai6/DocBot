FROM python:3.12-slim

WORKDIR /app

# Create non-root user for security
RUN useradd -m appuser

# DOCBOT-208: ODBC Driver 18 for Azure SQL / Microsoft Entra auth
# msodbcsql18 is only published for amd64 — skip on arm64 (Apple Silicon, Raspberry Pi)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    curl gnupg2 apt-transport-https ca-certificates unixodbc-dev \
 && ARCH="$(dpkg --print-architecture)" \
 && if [ "$ARCH" = "amd64" ]; then \
      curl -sSL https://packages.microsoft.com/keys/microsoft.asc \
        | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
      && echo "deb [arch=amd64 signed-by=/usr/share/keyrings/microsoft-prod.gpg] \
         https://packages.microsoft.com/debian/12/prod bookworm main" \
         > /etc/apt/sources.list.d/mssql-release.list \
      && apt-get update -y \
      && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18; \
    else \
      echo "Skipping msodbcsql18 — not available for $ARCH (Azure SQL unavailable on this build)"; \
    fi \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

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
