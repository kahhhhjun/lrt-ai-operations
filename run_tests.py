"""Unit tests — validated against hardcoded frequency tables and weather cap constants."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from core.calculator import (
    default_frequency, get_baseline_pax, WEATHER_MAX_FREQ, compute_options
)

passed = 0
failed = 0

def test(name, got, expected):
    global passed, failed
    ok = got == expected
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name}")
    print(f"         expected={expected!r}  got={got!r}")
    if ok: passed += 1
    else:  failed += 1

print("\n=== LRT AI — Unit Tests ===\n")

# Test 1: Weekday morning peak frequency (08:00, Monday)
test("Weekday 08:00 frequency",
     default_frequency(8, 0), 19)

# Test 2: Weekday evening peak frequency (18:00, Friday)
test("Weekday 18:00 frequency",
     default_frequency(18, 4), 20)

# Test 3: Saturday midday frequency (12:00)
test("Saturday 12:00 frequency",
     default_frequency(12, 5), 11)

# Test 4: Sunday morning frequency (10:00)
test("Sunday 10:00 frequency",
     default_frequency(10, 6), 6)

# Test 5: Weather cap — rainy (trains slow down)
test("Rainy weather max frequency cap",
     WEATHER_MAX_FREQ["rainy"], 17)

# Test 6: Weather cap — stormy (strictest cap)
test("Stormy weather max frequency cap",
     WEATHER_MAX_FREQ["stormy"], 12)

# Test 7: Baseline pax — weekday evening peak (18:00)
test("Weekday 18:00 baseline passengers",
     get_baseline_pax(18, 0), 12_000)

print(f"\n=== Results: {passed} passed, {failed} failed ===\n")
