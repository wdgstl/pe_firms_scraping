FROM python:3.10

# 1) Install curl (needed for Ollama installer)
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/*

# 2) Install Ollama CLI
RUN curl -fsSL https://ollama.com/install.sh | sh

# 3) Ensure Ollama listens on all interfaces (default port 11434)
ENV OLLAMA_HOST=0.0.0.0:11434

# 4) Start the daemon, wait a bit, pull the model, then shut it down—all in one layer
RUN ollama serve & \
    sleep 5 && \
    ollama pull mixtral && \
    pkill ollama

# 5) Set up your Python app
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY ./src ./src
COPY pefirms.csv .

# 6) Expose Ollama + your app port
EXPOSE 11434 8000

# 7) On container start, launch both Ollama and your Python service
CMD ["sh", "-c", "ollama serve & python3 -u ./src/main.py"]
