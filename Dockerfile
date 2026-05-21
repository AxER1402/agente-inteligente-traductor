FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Instalar dependencias del sistema necesarias para MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
# dataset.csv y modelo.pkl se incluyen en la imagen para persistencia
COPY . .

# Exponer puerto (Railway lo inyecta via $PORT)
EXPOSE 8000

# Arrancar el servidor FastAPI leyendo $PORT de Railway
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT:-8000}"]
