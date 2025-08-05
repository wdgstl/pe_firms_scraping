# Stage 1: ollama-base – pulls mixtral once
FROM python:3.10 AS ollama-base

# install curl + Ollama CLI
RUN apt-get update && \
    apt-get install -y curl && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSL https://ollama.com/install.sh | sh

# ensure Ollama listens on all interfaces
ENV OLLAMA_HOST=0.0.0.0:11434

# start → pull → stop in one layer
RUN ollama serve & \
    sleep 5 && \
    ollama pull mixtral && \
    pkill ollama

# Stage 2: app – builds your scraping service on top of ollama-base
FROM ollama-base AS app

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./src ./src
COPY pefirms.csv .

EXPOSE 11434 8000

CMD ["sh", "-c", "ollama serve & python3 -u ./src/main.py"]
