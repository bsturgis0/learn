FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    tesseract-ocr \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first
COPY requirements-render.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements-render.txt

# Copy the application code
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Run the application
CMD ["streamlit", "run", "src/app.py"]