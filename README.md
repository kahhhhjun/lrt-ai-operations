<div align="center">

```
██████╗  █████╗ ██╗██╗     ███╗   ███╗██╗███╗   ██╗██████╗
██╔══██╗██╔══██╗██║██║     ████╗ ████║██║████╗  ██║██╔══██╗
██████╔╝███████║██║██║     ██╔████╔██║██║██╔██╗ ██║██║  ██║
██╔══██╗██╔══██║██║██║     ██║╚██╔╝██║██║██║╚██╗██║██║  ██║
██║  ██║██║  ██║██║███████╗██║ ╚═╝ ██║██║██║ ╚████║██████╔╝
╚═╝  ╚═╝╚═╝  ╚═╝╚═╝╚══════╝╚═╝     ╚═╝╚═╝╚═╝  ╚═══╝╚═════╝
```

**AI-Powered LRT Operations Decision Support System**

*Kelana Jaya · Ampang · Sri Petaling*

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Z.AI GLM](https://img.shields.io/badge/Z.AI-GLM--5.1-6C63FF?style=flat-square)](https://z.ai)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?style=flat-square&logo=sqlite&logoColor=white)](https://sqlite.org)
[![License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)
[![UMHackathon](https://img.shields.io/badge/UMHackathon-2026-F59E0B?style=flat-square)](https://umhackathon.com)

</div>

---

## What is RAILMIND?

LRT duty managers in Malaysia face a recurring challenge: weather changes, live events, and emergencies can completely alter passenger demand — and right now, adapting the train schedule to these situations is done manually, takes 15–20 minutes, and depends entirely on individual experience.

**RAILMIND** eliminates that gap. Describe what is happening — type it, upload an event poster, or feed it a CCTV screenshot — and the system instantly computes three scheduling options and uses Z.AI GLM-5.1 to reason through them, pick the best one, and explain why in plain language. Apply it in one click. Save it so any shift can pick up exactly where the last one left off.

> **Without GLM-5.1, RAILMIND can still calculate the three options — but it cannot interpret free-text, reason across context, or explain its choice. The AI layer is what turns a calculator into a decision-support system.**

---

## Features

### Core Scheduling Engine
- **Full 18-hour schedule view** — 06:00 to 24:00, showing trains per hour, headway, load factor, operational cost, and carbon tax for every slot across all three lines
- **Three-option trade-off** — Conservative, Moderate, and Aggressive options computed in under 2 seconds using a deterministic rule engine grounded in real LRT ridership data
- **Daily and weekly cost breakdown** — operational cost, carbon tax, and grand total with an interactive weekly chart

### AI-Powered Inputs
- **Manual inputs** — weather, event type, crowd size, emergency type via dropdowns
- **Free-text description** — describe the situation in plain English; GLM-5-Turbo extracts weather, events, and emergency type automatically
- **Event poster upload** — upload a concert poster or flyer; OCR + GLM reads the event name, date, and time
- **CCTV crowd detection** — upload a platform screenshot; HuggingFace DETR counts people and estimates passengers per hour

### GLM Reasoning
- **Live streaming recommendation** — GLM-5.1 streams its reasoning in real time, citing load factor, headway, cost delta, and historical precedents before committing to a recommendation
- **Auto shift briefing** — after every schedule update, GLM writes a concise handover note for the outgoing shift

### Emergency Handling

| Emergency Type | System Response |
|---|---|
| Signal failure | Full service halt. No trains until the fault is cleared. |
| Power failure | Reduced service on backup power. |
| Overcrowding / stampede risk | Urgent frequency increase flagged immediately. |

> Safety overrides are **hard-coded in the math layer** and cannot be overridden by GLM, regardless of its reasoning.

### Persistence & Authentication
- **SQLite persistence** — every adjusted schedule is saved by `(date, line)` and reloads automatically on the next visit
- **Staff authentication** — registration and login with SHA-256 password hashing and session management

### Resilience
RAILMIND never leaves a duty manager without an answer, even when external APIs are unavailable:
1. **Streaming fails** → retries with a standard blocking call
2. **GLM unreachable** → math-based recommendation with a visible warning banner
3. **Text extraction fails** → keyword-based fallback extracts weather, events, and emergencies from common Malay/English terms

---

## Demo

> **Live deployment:** [Hosted on Streamlit Cloud](https://lrt-ai-operations-asaqmnlv7yfpy76tddwbdi.streamlit.app/) *(link in repo About section)*

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit 1.30+ |
| Language | Python 3.12 |
| AI Reasoning | Z.AI GLM-5.1 (streaming) |
| AI Extraction | Z.AI GLM-5-Turbo (fast) |
| Crowd Detection | HuggingFace DETR (facebook/detr-resnet-50) |
| Database | SQLite 3 |
| Charts | Altair |
| Data | pandas |
| HTTP | requests |
| Image Processing | Pillow |

---

## Project Structure

```
railmind/
├── app.py                    # Main Streamlit UI — schedule view, input modes, apply/save flow
├── auth.py                   # Authentication layer — registration, login, session management
├── test_glm.py               # GLM API connection tester — run this first
├── requirements.txt
├── start_authenticated.bat   # Windows: double-click to launch
│
├── core/
│   ├── calculator.py         # Rule engine — ridership baselines, weather multipliers,
│   │                         #   event profiles, emergency protocols, three-option computation
│   ├── recommender.py        # Orchestrator — GLM prompt building, streaming, shift briefing
│   ├── glm_client.py         # Z.AI GLM wrapper — streaming, extraction, DETR crowd counting
│   └── database.py           # SQLite CRUD — save, load, delete, list schedules
│
└── data/
    ├── history.json          # Past incidents used as GLM historical context
    ├── synthetic.csv         # Sample ridership data
    ├── generator.py          # Synthetic data generator (seed 42)
    └── lrt_schedules.db      # Auto-created on first run
```

---

## Getting Started

### Prerequisites
- Python 3.12 or later
- A Z.AI API key — get one at [z.ai](https://z.ai)
- *(Optional)* A HuggingFace API key for CCTV crowd detection — [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)

### 1. Clone the repository

```bash
git clone https://github.com/kahhhhjun/lrt-ai-operations.git
cd lrt-ai-operations/lrt_ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
GLM_API_KEY=your_z_ai_api_key_here
GLM_ENDPOINT=https://api.z.ai/api/anthropic/v1/messages
GLM_MODEL=glm-5.1
GLM_MODEL_FAST=glm-5-turbo
HF_API_KEY=your_huggingface_api_key_here   # optional
```

### 4. Test your GLM connection

```bash
python test_glm.py
```

Expected output:
```
=== GLM Connection Test ===
API Key  : SET (your_k...)
✅ SUCCESS! GLM replied: Hello! ...
```

### 5. Run the application

```bash
streamlit run auth.py
```

Then open [http://localhost:8501](http://localhost:8501) in your browser.

> **Windows shortcut:** double-click `start_authenticated.bat`

---

## How to Use

1. **Register and log in** — create a staff account on the authentication page
2. **Pick a date and line** — the standard 18-hour schedule loads automatically
3. **Set the time window** — drag the sliders to the hours affected by the situation
4. **Describe the situation** — choose from four input methods:
   - *Manual inputs* — dropdowns for weather, event type, emergency
   - *Describe in text* — type a free-text description, GLM extracts the details
   - *Upload image* — upload a concert poster, GLM reads the event details via OCR
   - *CCTV crowd detection* — upload a platform screenshot, DETR counts the crowd
5. **Click Analyse** — three option cards appear instantly with load factor, cost, and passengers served
6. **Review GLM reasoning** — GLM streams its recommendation and explanation in real time
7. **Apply and save** — choose an option (or override GLM's pick), click Apply, then Save

Previously saved schedules reload automatically when you revisit the same date and line.

---

## Streamlit Cloud Deployment

RAILMIND is deployed on Streamlit Community Cloud. For Streamlit Cloud, replace the `.env` file with Streamlit Secrets:

1. Go to your app on [share.streamlit.io](https://share.streamlit.io) → **Settings** → **Secrets**
2. Add your credentials in TOML format:

```toml
GLM_API_KEY     = "your_z_ai_api_key_here"
GLM_ENDPOINT    = "https://api.z.ai/api/anthropic/v1/messages"
GLM_MODEL       = "glm-5.1"
GLM_MODEL_FAST  = "glm-5-turbo"
HF_API_KEY      = "your_huggingface_api_key_here"
```

3. Update `core/glm_client.py` to read from `st.secrets` with a fallback to `os.getenv()`:

```python
import streamlit as st

def _secret(key, default=None):
    try:
        return st.secrets[key]
    except (KeyError, AttributeError, FileNotFoundError):
        return os.getenv(key, default)

GLM_API_KEY = _secret("GLM_API_KEY")
# ... and so on for the other variables
```

> **Note:** The SQLite database and user accounts are stored on Streamlit Cloud's ephemeral filesystem and will reset on redeployment. This is expected behaviour for the free tier.

---

## Configuration Reference

| Variable | Description | Default |
|---|---|---|
| `GLM_API_KEY` | Z.AI API key | — (required) |
| `GLM_ENDPOINT` | Z.AI completions endpoint | `https://api.z.ai/api/anthropic/v1/messages` |
| `GLM_MODEL` | Primary model for reasoning and briefings | `glm-5.1` |
| `GLM_MODEL_FAST` | Fast model for text/image extraction | `glm-5-turbo` |
| `HF_API_KEY` | HuggingFace key for CCTV detection | — (optional) |

---

## Roadmap

- [ ] Connect to live RapidKL gate count data — replace synthetic ridership baselines
- [ ] Multi-line coordination — adjust all three lines simultaneously for interchange stations
- [ ] Push alerts to station staff when a schedule changes
- [ ] Outcome feedback loop — log real ridership vs predicted to improve GLM accuracy over time
- [ ] Migrate SQLite to PostgreSQL with a normalised schema for queryable history
- [ ] Vector database for historical precedent retrieval (pgvector or FAISS)
- [ ] Role-based access control — duty managers, supervisors, and system administrators
- [ ] bcrypt password hashing to replace SHA-256

---

## Built For

**UMHackathon 2026** — Domain: AI for Economic Empowerment & Decision Intelligence

Submitted under the track requiring Z.AI GLM as the core reasoning engine.

---

## License

[MIT](LICENSE)

---

<div align="center">

*RAILMIND — because every train that runs on time is a decision that went right.*

</div>
