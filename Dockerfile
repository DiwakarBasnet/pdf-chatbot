# Use miniforge3 base image
FROM condaforge/miniforge3:latest

# Set environment variables
ENV HOST=0.0.0.0
ENV LISTEN_PORT=8080
ENV HF_ENDPOINT=https://hf-mirror.com

# Expose the application port
EXPOSE 8080

# Update the package list and install git, curl
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    openssl \
    git \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Create a directory named files
RUN mkdir -p /app/files

# Copy conda environment file to the container
COPY environment.yml .

# Update conda, create conda environment, and install conda packages
RUN conda update -n base -c conda-forge conda && \
    conda env create -f environment.yml && \
    conda clean -afy

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "langchain", "/bin/bash", "-c"]

# Copy and install pip requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the application code to the container
COPY model.py streamlit_app.py rag_util.py .env ./

# Activate the environment and run the application
CMD ["conda", "run", "--no-capture-output", "-n", "langchain", "streamlit", "run", "streamlit_app.py", "--server.port", "8080", "--server.address", "0.0.0.0"]
