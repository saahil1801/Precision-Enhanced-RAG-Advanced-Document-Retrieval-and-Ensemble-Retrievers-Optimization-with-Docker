# Use the official lightweight Python image compatible with ARM architecture.
FROM --platform=linux/arm64 python:3.10-slim

# Set environment variables to ensure python outputs everything to the console.
ENV PYTHONUNBUFFERED=1

# Install system dependencies and build tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory.
WORKDIR /app

# Copy the requirements file to the working directory.
COPY requirements.txt .

# Install the Python dependencies.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code to the working directory.
COPY . .

# Expose the port Streamlit will run on.
EXPOSE 8501
# Command to run the Streamlit app.
CMD ["streamlit", "run", "main.py"]
