# Use slim Python image to keep the container small
FROM python:3.11-slim

WORKDIR /app

# Copy dependency definition first — Docker caches this layer.
# If only code changes (not dependencies), pip install is skipped on rebuild.
COPY pyproject.toml .

# Install app dependencies + the web server
RUN pip install --no-cache-dir \
    langgraph>=1.1 \
    langchain-core>=1.2 \
    langchain-groq>=1.1 \
    pydantic>=2.0 \
    fastapi>=0.110 \
    uvicorn>=0.29

# Copy application code after dependencies — keeps code-change rebuilds fast
COPY fraud_agent/ fraud_agent/

# GROQ_API_KEY is injected at runtime via Kubernetes Secret (not baked in here)
ENV PYTHONUNBUFFERED=1

EXPOSE 8080

CMD ["uvicorn", "fraud_agent.server:app", "--host", "0.0.0.0", "--port", "8080"]
