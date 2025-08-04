FROM python:3.10-slim AS builder

WORKDIR /app

# ติดตั้ง system deps ที่จำเป็นสำหรับ native build
RUN apt-get update && apt-get install -y \
    build-essential \
    libffi-dev \
    libssl-dev \
    cargo \
    rustc \
    git \
    libzstd-dev \
    libjpeg-dev \
    patchelf \             
    python3-dev \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements-docker.txt .

RUN pip install --upgrade pip
RUN pip install -r requirements-docker.txt

COPY . .

# Compile with Nuitka
RUN pip install nuitka \
 && nuitka --onefile --output-filename=main app.py

# ---- Minimal runtime ----
FROM debian:bullseye-slim

WORKDIR /app

RUN apt-get update && apt-get install -y libstdc++6 libzstd1 libssl1.1 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=builder /app/main /app/main
RUN chmod +x /app/main

EXPOSE 8000
CMD ["./main"]