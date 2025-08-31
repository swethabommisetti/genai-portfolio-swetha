# 🧠 GenAI Portfolio – Swetha

This project demonstrates a **production-ready, Dockerized GenAI agent environment** with **multi-environment support (dev/qa/prod)**.
It showcases **LLM-powered agents** (Receipt Scanner, Book Recommender) with **end-to-end architecture**: frontend (Streamlit), orchestration (LangChain/LangGraph), and backend integration (Supabase, Doppler, LangSmith).

---

## 🚀 Features

* ✅ **Agent-based AI apps** (LangChain, OCR, Streamlit UI)
* ✅ **Multi-LLM backends**: Groq (speed), Mistral (open-source), OpenAI-ready (accuracy)
* ✅ **Docker-first workflow** – dev, qa, prod isolation
* ✅ **Environment configs** via `.env` + build args
* ✅ **Supabase integration** – Postgres, auth, storage
* ✅ **Secrets management** – Doppler + `.streamlit/secrets.toml`
* ✅ **CI/CD ready** – GitHub Actions + Streamlit Cloud deploy
* ✅ **Secure login (optional)** for recruiter demos
* ✅ **iPhone/mobile testing** on local WiFi
* ✅ **Observability** – LangSmith run traces & evaluation

---

## 🏗️ Architecture

  1. Receipt Scanner
     
<img width="1024" height="1536" alt="ChatGPT Image Aug 29, 2025, 11_22_25 PM" src="https://github.com/user-attachments/assets/2ea56fe7-3d22-4782-9a3e-95c758010aa4" />


**Flow**

1. Streamlit UI (`Home.py`, modular `pages/`)
2. Secrets via Doppler → Streamlit Cloud → Env
3. Agents orchestrated with LangChain/LangGraph
4. LLM inference via Groq/Mistral/OpenAI
5. Data stored in Supabase (visitors, receipts, logs)
6. Optional LangSmith for tracing/evaluation

---

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
│   ├── Home.py
│   ├── agents/
│   │   ├── pages/
│   │   └── receipt_extractor/
│   └── utils/
├── docs/
│   └── architecture.png
└── .github/
    └── workflows/
```

---

## 🧪 Local Dev (iPhone Testing)

```bash
docker build -t genai-portfolio-swetha:dev --build-arg ENV=dev .
docker run -p 8501:8501 --env-file=./environments/dev/.env genai-portfolio-swetha:dev
```

* Desktop: [http://localhost:8501](http://localhost:8501)
* iPhone (same WiFi): `http://192.168.x.x:8501`

---

## 🌐 Public Demo (Recruiters)

Deployed on **Streamlit Cloud**:
👉 [https://genai-portfolio-swetha.streamlit.app](https://genai-portfolio-swetha.streamlit.app)

(Private login available on request.)

---

## 🧱 Goals

* Build **real-world GenAI apps** with container-first architecture
* Showcase **modular, production-style AI agents**
* Demonstrate **cloud-native workflows** (Supabase + Doppler + Streamlit)
* Learn & apply **multi-agent orchestration + observability**

---

## 💡 Skills Highlighted

* LLM orchestration (LangChain + LangGraph)
* Multi-LLM strategy (Groq, Mistral, OpenAI)
* Database integration (Supabase)
* Secrets management (Doppler, TOML)
* Containerization (Docker multi-env)
* CI/CD with GitHub Actions + Streamlit Cloud
* Evaluation & tracing (LangSmith)
* Recruiter-ready UI/UX with modular Streamlit pages

---

## 📧 Contact

📬 Best method: listed in Resume / LinkedIn.

