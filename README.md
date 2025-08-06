# 🧠 GenAI Portfolio - Swetha

This project demonstrates a fully Dockerized, production-ready GenAI agent environment with multi-environment support (`dev`, `qa`, `prod`). Designed to showcase agentic AI apps and streamline the interview/demo experience.

## 🚀 Features

- ✅ Agent-based AI app architecture (LangChain, OCR, Streamlit)
- ✅ Docker-based development and isolation
- ✅ Environment-specific configuration via `.env` and build args
- ✅ Secure login-protected Streamlit interface (for recruiters)
- ✅ Easy iPhone/mobile testing over home network

## 📦 Repo Structure

```
genai-portfolio-swetha/
├── Dockerfile
├── requirements.txt
├── requirements.dev.txt
├── environments/
│   ├── dev/.env
│   ├── qa/.env
│   └── prod/.env
├── src/
│   └── agents/
│       ├── helloworld.py
│       └── pages/
├── docs/
│   └── SETUP_NOTES.md
├── .github/
│   └── workflows/ (CI/CD for formatting or future deploys)
├── README.md
```

## 🧪 Local Dev (iPhone Testing on Same WiFi)

```bash
docker build -t genai-portfolio-swetha:dev --build-arg ENV=dev .
docker run -p 8501:8501 --env-file=./environments/dev/.env genai-portfolio-swetha:dev
```

Then access the app at:
- Desktop: http://localhost:8501
- iPhone (same WiFi): http://192.168.x.x:8501

## 🌐 Public Demo for Recruiters

Once ready, the prod branch can be deployed to **Streamlit Cloud** and shared as a link (with temporary login):

```
https://swetha-genai.streamlit.app
```

> 💬 Recruiters can reach out for access credentials.

---

## 🧱 Goals

- Build real-world GenAI apps with full control
- Showcase AI agents like **Receipt Scanner** and **Book Recommender**
- Learn & demonstrate container-based AI workflows

---

## 📧 Contact

Swetha (AI/ML Engineer in transition) — _Reach out via GitHub or LinkedIn for demo credentials_
