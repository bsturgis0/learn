FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements-render.txt .

# Install Python dependencies
RUN pip install -r requirements-render.txt

# Copy the rest of the application
COPY . .

# Expose port 8501 for Streamlit
EXPOSE 8501

# Set environment variables
ENV AWS_ACCESS_KEY_ID=your_aws_access_key
ENV AWS_SECRET_ACCESS_KEY=your_aws_secret_key
ENV AWS_REGION=your_aws_region
ENV GOOGLE_API_KEY=your_google_api_key
ENV TAVILY_API_KEY=your_tavily_api_key

# Run the application
CMD ["streamlit", "run", "src/app.py"]