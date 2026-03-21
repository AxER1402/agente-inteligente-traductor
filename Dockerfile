FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Instalar dependencias del sistema necesarias para OpenCV y MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libxcb-cursor0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Comando por defecto para correr el detector de gestos
CMD ["python", "predecir_senia.py"]
