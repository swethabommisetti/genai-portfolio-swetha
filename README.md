# GenAI Portfolio – Swetha

This project demonstrates a **production-ready, Dockerized GenAI agent environment** with **multi-environment support (dev/qa/prod)**.
It showcases **LLM-powered agents** (Receipt Scanner, Book Recommender) with **end-to-end architecture**: frontend (Streamlit), orchestration (LangChain/LangGraph), and backend integration (Supabase, Doppler, LangSmith).

## Features

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

##  Architecture

  1. Receipt Scanner
     
**Flow**

1. Streamlit UI (`Home.py`, modular `pages/`)
2. Secrets via Doppler → Streamlit Cloud → Env
3. Agents orchestrated with LangChain/LangGraph
4. LLM inference via Groq/Mistral/OpenAI
5. Data stored in Supabase (visitors, receipts, logs)
6. Optional LangSmith for tracing/evaluation

## Repo Structure

```
genai-portfolio-swetha/
│
├── .streamlit/                              → Streamlit app settings (theme, secrets)
│   └── config.toml
│
├── src/                                     → Main application code
│   ├── bookrecommender_homepage.py          → Entry page for the Book Recommender
│   ├── portfolio_homepage.py                → Portfolio overview page
│   ├── receiptscanner_homepage.py           → Entry page for the Receipt Scanner
│   │
│   ├── agents/                              → AI agents (brains of the app)
│   │   ├── analytics/                       → Agents that analyze data
│   │   └── receipt_extractor/               → AI agent that extracts info from receipts
│   │       └── agent.py
│   │
│   ├── pages/                               → Streamlit app pages (UI for each feature)
│   │   ├── 1_receiptscanner_run.py                   → Run Receipt Scanner
│   │   ├── 3_bookrecommender_run.py                  → Run Book Recommender
│   │   ├── 12_receiptscanner_documentation.py        → Docs for Receipt Scanner
│   │   ├── 13_receiptscanner_analytics.py            → Analytics for receipts
│   │   ├── 14_evaluator_home.py                      → Evaluator dashboard
│   │   ├── 15_evaluator_manual_scoring.py            → Manual scoring of AI outputs
│   │   ├── 16_evaluator_error_tagging.py             → Tag errors in AI outputs
│   │   ├── 17_utils_home.py                          → Utilities hub page
│   │   ├── 18_evaluator_header_accuracy.py           → Checks header accuracy
│   │   ├── 19_utils_altered_images.py                → Create altered/perturbed images
│   │   ├── 20_evaluator_altered_images.py            → Test AI on altered images
│   │   ├── 21_evaluator_consistency.py               → Check if AI is consistent
│   │   ├── 22_evaluator_latency.py                   → Measure AI speed (latency)
│   │   ├── 32_bookrecommender_documentation.py       → Docs for Book Recommender
│   │   └── 33_bookrecommender_analytics.py           → Analytics for book recs
│   │
│   ├── provisioning/           → Shared utilities for menus, themes, navigation
│   │   ├── api.py              → API bootstrap
│   │   ├── autostart_api.py    → Auto start logic
│   │   ├── config.py           → Config management
│   │   ├── docs.py             → Documentation helpers
│   │   ├── menu.py             → Sidebar menus
│   │   ├── nav.py              → Navigation handling
│   │   ├── theme.py            → App styling and colors
│   │   └── ui.py               → Reusable UI components
│   │
│   ├── tools/                  → Core tools (receipt parsing logic)
│   │   ├── receipt_extraction.py → Logic to extract data from receipts
│   │   └── receipt_schema.py   → Defines receipt fields and structure
│   │
│   └── utils/                  → Helper utilities
│       ├── email_utils.py      → Email integration
│       ├── evals_repo.py       → Stores evaluation data
│       ├── receipts_repo.py    → Stores receipt data
│       ├── supabase_utils.py   → Connects app with Supabase database
│       ├── tracking.py         → Logs and tracks app usage
│       └── visitor_service.py  → Tracks visitor analytics
│
├── Dockerfile                 → Docker setup for containerized deployment
├── docker-compose.yml         → Multi-service orchestration (if needed)
├── dockerignore               → Files ignored in Docker builds
├── README.md                  → Project introduction and instructions
├── requirements.txt           → Dependencies (for running the app)

```
## 🧪 Local Dev (iPhone Testing)

```bash
docker build -t genai-portfolio-swetha:dev --build-arg ENV=dev .
docker run -p 8501:8501 --env-file=./environments/dev/.env genai-portfolio-swetha:dev
```

* Desktop: [http://localhost:8501](http://localhost:8501)
* iPhone (same WiFi): `http://192.168.x.x:8501`

## 🌐 Public Demo (Recruiters)

Deployed on **Streamlit Cloud**:
👉 [https://genai-portfolio-swetha.streamlit.app](https://genai-portfolio-swetha.streamlit.app)

(Private login available on request.)

## 🧱 Goals

* Build **real-world GenAI apps** with container-first architecture
* Showcase **modular, production-style AI agents**
* Demonstrate **cloud-native workflows** (Supabase + Doppler + Streamlit)
* Learn & apply **multi-agent orchestration + observability**

## 💡 Skills Highlighted

* LLM orchestration (LangChain + LangGraph)
* Multi-LLM strategy (Groq, Mistral, OpenAI)
* Database integration (Supabase)
* Secrets management (Doppler, TOML)
* Containerization (Docker multi-env)
* CI/CD with GitHub Actions + Streamlit Cloud
* Evaluation & tracing (LangSmith)
* Recruiter-ready UI/UX with modular Streamlit pages


## 📧 Contact

📬 Best method: listed in Resume / LinkedIn.

