# SETUP_NOTES.md 🛠️

This project demonstrates how to **build and deploy agentic GenAI applications inside Docker containers** using Docker Compose,
with multi-environment support (`dev`, `qa`, `prod`), and a polished Streamlit UI accessible both locally (e.g., iPhone on home Wi-Fi)
and publicly (for recruiters).

---

## 🎯 Goal

- Build and run GenAI agents inside a **containerized development environment**
- Access the app on **iPhone via local network** during development
- Deploy to **Streamlit Cloud** once production-ready
- Share a **public demo URL with temp login** in my resume for recruiters to explore
- Maintain **full control** over environment visibility and codebase maturity

---

## ✅ Step 1: Docker + WSL2 Setup

- Installed **Docker Desktop on Windows**
- Enabled **WSL2 backend** for better networking and performance on Windows
- Verified that Docker builds and runs Python containers successfully

---

## ✅ Step 2: Dockerfile

The Dockerfile is designed as an **infrastructure-as-code recipe**:

- Uses `python:3.10-slim`
- Adds system dependencies (e.g., libgl1, git)
- Supports multi-environment via `--build-arg ENV=dev`
- Loads environment-specific configs from `/environments/<env>/`
- Supports both Streamlit (`port 8501`) and JupyterLab (`port 8888`)

---

## ✅ Step 3: Requirements Files

| File                   | Purpose                                                    |
| ---------------------- | ---------------------------------------------------------- |
| `requirements.txt`     | Base dependencies for all environments                     |
| `requirements.dev.txt` | Dev-only tools: `pytest`, `black`, `mypy`, `ipython`, etc. |

---

## ✅ Step 4: Folder Structure

Manually created using CMD:

```cmd
mkdir environments\dev
mkdir environments\qa
mkdir environments\prod
mkdir src\agents
mkdir docs
mkdir .github\workflows
type nul > .env.dev
type nul > .env.qa
type nul > .env.prod
```

Final structure:

```
genai-portfolio-swetha/
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements.dev.txt
├── .env.dev / .env.qa / .env.prod
├── environments/
│   ├── dev/.env
│   ├── qa/.env
│   └── prod/.env
├── src/
│   └── agents/
│       ├── Home.py
│       ├── pages/
│       │   ├── 1_📸_Receipt_Scanner.py
│       │   └── 2_📚_Book_Recommender.py
│       ├── utils/
│       │   ├── ocr_utils.py
│       │   └── recommender.py
│       └── __init__.py
├── .github/
│   └── workflows/
│       └── ci-cd.yml
├── docs/
│   └── SETUP_NOTES.md
```

---

## ✅ Step 5: Use Docker Compose (instead of docker run)

Use `docker-compose` to run **Streamlit**, **Jupyter**, or both.

### ➕ Build Images (first time)

```bash
docker-compose build
```

### ▶️ Run Streamlit only (for iPhone access)

```bash
docker-compose up streamlit
```

- Access via browser: http://localhost:8501
- Or from iPhone: http://<your-local-IP>:8501

### 🧠 Find IP (Windows):
```bash
ipconfig
# Look for IPv4 Address (e.g., 192.168.1.101)
```

### ▶️ Run Jupyter only (for notebooks)

```bash
docker-compose up jupyter
```

Access via browser:

```
http://127.0.0.1:8888/lab?token=<your-token>
```

### 🌀 Run Both Services

```bash
docker-compose up
```

### ❌ Stop Containers

```bash
docker-compose down
```

---

## ✅ Step 6: CI/CD Ready (optional)

- GitHub Actions under `.github/workflows/ci-cd.yml` can automate:
  - Linting
  - Pytest
  - Auto-deploy to Streamlit Cloud (or HuggingFace Spaces if desired)
- Only push `prod` branch for public deployment

---

## 💡 Bonus: Public URL + Login for Recruiters

Once the app is ready:

1. Push to GitHub (main/prod branch)
2. Connect to Streamlit Cloud
3. Deploy public demo:

```
https://swetha-genai.streamlit.app
```

4. Enable password login using `streamlit-authenticator`:
   - Temp username/password for each recruiter
   - Share creds only on request

