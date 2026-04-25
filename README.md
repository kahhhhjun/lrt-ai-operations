# LRT AI Operations — Decision Support System

> An AI-powered scheduling tool for Malaysian LRT duty managers. Describe the situation, get three options, let GLM-5.1 pick the best one — then apply and save in one click.

**Powered by Z.AI GLM-5.1 · Kelana Jaya · Ampang · Sri Petaling**

---

## Presentation & Demonstration Link
Link: https://drive.google.com/drive/folders/1kCi2BLf8Ktk48mooGZPc5qrzqo6rzX7O?usp=sharing


## What It Does

LRT duty managers deal with three things that can change the whole day — **weather**, **events**, and **emergencies**. Right now, adjusting the train schedule for these situations is done manually, takes 15–20 minutes, and depends on each individual's experience.

This system fixes that. Staff describe what is happening, the math engine instantly calculates three scheduling options (Conservative, Moderate, Aggressive), and GLM-5.1 picks the best one with a plain-language explanation. Staff apply it with one click and save it to the database for any shift to retrieve later.

**Without GLM-5.1, the system can still calculate the three options — but it cannot interpret free-text, cannot reason across context and history, and cannot explain its choice. GLM is essential.**

---

## Features

- **Full day schedule view** — 06:00 to 24:00, showing trains per hour, load factor, headway, and cost for every hour slot
- **Two input modes** — manual form (dropdowns) or free-text description
- **Instant three-option trade-off** — Conservative, Moderate, and Aggressive computed in under 2 seconds with full metrics
- **Live GLM reasoning** — GLM-5.1 streams its pick and explanation in real time, citing load factor, cost, and historical precedents
- **Six emergency types** — track incident, signal failure, power failure, train breakdown, evacuation, overcrowding — each with its own response logic and a recovery curve for the hours after clearance
- **Auto shift briefing** — GLM writes a handover note under 220 words after every schedule update
- **SQLite persistence** — every adjusted schedule is saved by (date, line) and can be reloaded by any shift at any time
- **Three-layer fallback** — if GLM is unavailable, the system retries, then falls back to a math-based recommendation, then to a keyword extractor — always keeping staff unblocked

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit 1.30+ |
| Core logic | Python 3.12 |
| AI reasoning | Z.AI GLM-5.1 (`ilmu-glm 5.1`) |
| Database | SQLite |
| Data | pandas |

---

## Project Structure

```
lrt_ai/
├── app.py                  # Streamlit UI — schedule view, input modes, apply/save flow
├── requirements.txt
├── .env                    # API keys (not committed)
├── test_glm.py             # GLM connection tester
├── core/
│   ├── calculator.py       # Math layer — ridership baselines, weather multipliers,
│   │                       # event profiles, three-option computation, emergency logic
│   ├── recommender.py      # Orchestrator — calls math core + GLM, builds prompts,
│   │                       # parses RECOMMENDATION tag, generates shift briefing
│   ├── glm_client.py       # Z.AI GLM wrapper — streaming, 
│   └── database.py         # SQLite CRUD — save, load, delete, list schedules
└── data/
    ├── generator.py        # Synthetic ridership CSV generator (seed 42)
    ├── synthetic.csv       # Generated sample data
    ├── history.json        # Past incidents used as historical context for GLM
    └── lrt_schedules.db    # SQLite database (auto-created on first run)
```

---

## Getting Started

### 1. Clone the repo

```bash
git clone https://github.com/kahhhhjun/lrt-ai-operations.git
cd lrt_ai
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root:

```env
GLM_API_KEY=your_z_ai_api_key_here
GLM_ENDPOINT=https://api.z.ai/api/paas/v4/chat/completions
GLM_MODEL=ilmu-glm-5.1
```

> Get your Z.AI API key at [z.ai](https://z.ai)

### 4. Test your GLM connection

```bash
python test_glm.py
```

You should see `✅ SUCCESS! GLM replied: ...`

### 5. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## How to Use

1. **Pick a date and line** — the standard 18-hour schedule loads automatically
2. **Set the time window** — drag the sliders to the hours affected by the situation
3. **Describe the situation** — choose manual inputs, type in free text, or upload an image
4. **Click Analyse** — three option cards appear instantly; GLM streams its reasoning
5. **Apply and save** — choose an option (or override GLM's pick), click Apply, then Save

Previously saved schedules are shown in the history panel and reload automatically when you revisit the same date and line.

---

## Emergency Handling

| Type | System Response |
|---|---|
| Track incident | Full suspension (0 trains). Recovery at max frequency after clearance. |
| Signal failure | Full service halt. No trains until fault is cleared. |
| Power failure | Reduced service on backup power. System shows what is safe to run. |
| Train breakdown | Minor reduction — one train fewer per hour. |
| Evacuation | Maximum frequency — get everyone out fast. |
| Overcrowding | Urgent frequency increase flagged immediately. |

**Safety overrides are hard-coded in the math layer and cannot be changed by GLM.** Even if GLM reasons otherwise, a track incident always results in 0 trains during the active window.

---

## Fallback Behaviour

The system never leaves staff without an answer, even when GLM is down:

1. **Streaming fails** → retries with a standard blocking call
2. **GLM unreachable** → shows a math-based recommendation with a yellow warning banner
3. **Extraction fails** (free text or image) → keyword-based fallback extracts weather, emergency type, and event from common Malay/English terms

All fallback paths are clearly labelled in the UI so staff always know what kind of recommendation they are reading.

---

## Configuration

All settings are loaded from `.env` via `python-dotenv`. No values are hard-coded.

| Variable | Description | Default |
|---|---|---|
| `GLM_API_KEY` | Your Z.AI API key | — |
| `GLM_ENDPOINT` | Z.AI completions endpoint | `https://api.z.ai/api/paas/v4/chat/completions` |
| `GLM_MODEL` | Model name | `glm-4` |

---

## Running Tests

```bash
# Run all unit tests
pytest -q

# Run with coverage report
pytest --cov=core --cov-report=term-missing
```

Tests mock all external calls (GLM API, OCR.space) so they run offline. The database tests use an in-memory SQLite instance.

---

## Roadmap

- [ ] Connect to live RAPID gate count data (replace synthetic baselines)
- [ ] Multi-line coordination — adjust all three lines together for interchange stations
- [ ] Push alerts to station staff when a schedule changes
- [ ] Outcome feedback loop — log real ridership vs predicted to improve GLM accuracy over time
- [ ] Migrate SQLite to Postgres with normalized schema for queryable history
- [ ] Vector database for historical precedent retrieval (pgvector or FAISS)

---

## Built For

**UMHackathon 2026** — Domain: AI for Economic Empowerment & Decision Intelligence

> Submitted under the track requiring Z.AI GLM as the core reasoning engine.

---

## License

MIT
