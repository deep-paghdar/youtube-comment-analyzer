# Use a Python base image
FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install NLTK and download the punkt_tab dataset
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('stopwords')"

# Copy the rest of the project files
COPY . .

# Expose the app port
EXPOSE 5000

# Run the application
CMD ["python", "run.py"]
