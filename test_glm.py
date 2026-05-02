"""Integration tests — requires live credentials in .env"""
import sys, os, requests
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv
load_dotenv()

passed = 0
failed = 0

def test(name, ok, detail=""):
    global passed, failed
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    if detail:
        print(f"         {detail}")
    if ok: passed += 1
    else:  failed += 1

print("\n=== LRT AI — Integration Tests ===\n")


# ── Test 1 & 2: GLM API connectivity ─────────────────────────────────────────
print("1. GLM API connectivity")
try:
    from core.glm_client import GLM_API_KEY, GLM_ENDPOINT, GLM_MODEL
    payload = {
        "model": GLM_MODEL,
        "max_tokens": 50,
        "messages": [{"role": "user", "content": "Reply with the single word: OK"}],
    }
    resp = requests.post(
        GLM_ENDPOINT,
        headers={"x-api-key": GLM_API_KEY, "anthropic-version": "2023-06-01",
                 "content-type": "application/json"},
        json=payload, timeout=20,
    )
    ok = resp.status_code == 200
    body = resp.json()
    reply = body.get("content", [{}])[0].get("text", "").strip()
    test("GLM HTTP 200 response", ok, f"status={resp.status_code}")
    test("GLM model reply parsed correctly", bool(reply), f"model reply: '{reply}'")
except Exception as e:
    test("GLM HTTP 200 response", False, str(e))
    test("GLM model reply parsed correctly", False, str(e))


# ── Test 3, 4, 5: SQLite DB save / load / delete cycle ───────────────────────
print("\n2. SQLite DB integration")
try:
    from core.database import init_db, save_schedule, load_schedule, delete_schedule
    init_db()
    _test_date = "2099-01-01"
    _test_line = "Kelana Jaya"
    _test_sched = [{"hour": 8, "time_slot": "08:00-09:00", "standard_frequency": 19,
                    "recommended_frequency": 20, "extra_trains": 1,
                    "standard_cost_rm": 6650, "extra_cost_rm": 350, "total_cost_rm": 7000,
                    "standard_carbon_tax_rm": 85.5, "extra_carbon_tax_rm": 4.5,
                    "total_carbon_tax_rm": 90.0, "expected_passengers_per_hr": 11000,
                    "load_factor_pct": 68.75, "standard_load_factor_pct": 72.4,
                    "passengers_served_per_hr": 11000, "headway_std_min": 3,
                    "headway_rec_min": 3, "has_event": False, "is_event_tail": False,
                    "event_names": [], "em_status": None}]
    save_schedule(_test_date, _test_line, _test_sched, weather="rainy",
                  events=[], total_std_cost=6650, total_extra_cost=350, total_cost=7000)
    loaded = load_schedule(_test_date, _test_line)
    test("DB save schedule", loaded is not None, f"record written for {_test_date}")
    load_ok = loaded is not None and loaded["weather"] == "rainy" and len(loaded["schedule"]) == 1
    test("DB load schedule (data integrity)", load_ok,
         f"weather='{loaded['weather'] if loaded else '?'}', rows={len(loaded['schedule']) if loaded else 0}")
    delete_schedule(_test_date, _test_line)
    test("DB delete schedule", load_schedule(_test_date, _test_line) is None, "test record cleaned up")
except Exception as e:
    test("DB save schedule", False, str(e))
    test("DB load schedule (data integrity)", False, str(e))
    test("DB delete schedule", False, str(e))


# ── Test 6 & 7: HuggingFace DETR crowd detection ─────────────────────────────
print("\n3. HuggingFace DETR crowd detection")
try:
    from core.glm_client import HF_API_KEY
    from PIL import Image
    import io
    # Generate a proper 200x200 grey JPEG — valid input for DETR
    _img = Image.new("RGB", (200, 200), color=(100, 100, 100))
    _buf = io.BytesIO()
    _img.save(_buf, format="JPEG")
    _tiny_jpg = _buf.getvalue()
    hf_url = "https://router.huggingface.co/hf-inference/models/facebook/detr-resnet-50"
    r = requests.post(
        hf_url,
        headers={"Authorization": f"Bearer {HF_API_KEY.strip()}",
                 "Content-Type": "image/jpeg"},
        data=_tiny_jpg, timeout=30,
    )
    api_ok = r.status_code == 200
    test("HF API HTTP 200 response", api_ok, f"status={r.status_code}")
    detections = r.json() if api_ok else []
    parsed_ok = isinstance(detections, list)
    person_count = sum(1 for d in detections
                       if d.get("label", "").lower() == "person" and d.get("score", 0) >= 0.5)
    test("HF response parsed as detection list", parsed_ok,
         f"detections={len(detections)}, persons={person_count} (blank image -> 0 expected)")
except Exception as e:
    test("HF API HTTP 200 response", False, str(e))
    test("HF response parsed as detection list", False, str(e))


print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
