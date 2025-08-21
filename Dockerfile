# syntax=docker/dockerfile:1
FROM python:3.11-slim

# Faster startup + live logs
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install only what you need for your code
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source
COPY . .

# Run your app
CMD ["python", "app.py"]