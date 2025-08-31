# ---------- Base ----------
FROM python:3.10-slim

# Build args / envs
ARG ENV=dev
ENV ENV_MODE=${ENV} \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    PYTHONPATH=/app/src

WORKDIR /app

# ---------- OS deps (lean) ----------
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential git \
  && rm -rf /var/lib/apt/lists/*

# ---------- Python deps ----------
COPY requirements.txt .
RUN pip install -r requirements.txt

# ---------- App code ----------
# (keep your environment-specific files)
COPY environments/${ENV_MODE} /app/environments
COPY . /app

# ---------- Streamlit ----------
# Streamlit listens on 8501
EXPOSE 8501

# Optional: create a minimal Streamlit config (quiet logs)
RUN mkdir -p /root/.streamlit && \
    printf "[server]\nheadless = true\n" > /root/.streamlit/config.toml

# ---------- Run the app ----------
CMD ["streamlit", "run", "src/portfolio_homepage.py", "--server.port=8501", "--server.address=0.0.0.0"]
