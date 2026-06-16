# Python 3.11 (not 3.14) — scikit-learn / xgboost have mature wheels here.
FROM python:3.11-slim

# Keep Python output unbuffered and skip .pyc files in the image.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install dependencies first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy the rest of the project.
COPY . .

# Both services use this image; docker-compose overrides CMD per service.
EXPOSE 8000 8501

# Default: run the API. (docker-compose overrides this for the dashboard.)
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
