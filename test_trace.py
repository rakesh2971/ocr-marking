"""Quick trace: test repair_merged_token logic and grep for where 2.720.5 enters."""
import re

def repair_merged_token(text):
    t = text.strip()
    if not t:
        return None
    # 1. S0/S00 diameter repair
    t = re.sub(r'^S0{1,2}(\d)', '\u2300\\1', t)
    t = re.sub(r'^[Ss][Oo0](\d)', '\u2300\\1', t)
    # 2a. Decimal + datum fused split
    datum_fused = re.match(r'^([\u2300\u00d8\u00f8\u00d4]?[\d]+\.\d+)([A-Z]{1,3})$', t)
    if datum_fused:
        t = f"{datum_fused.group(1)} {datum_fused.group(2)}"
    # 2b. findall double-decimal split
    decimal_parts = re.findall(r'[SR\u2300\u00d8\u00f8\u00d4\u00b1]?\d+\.\d+', t)
    if len(decimal_parts) >= 2:
        first = decimal_parts[0]
        print(f"  SPLIT: '{t}' -> kept '{first}'")
        t = first
    return t

tests = ['2.720.5', '57.70.07', '0.05.5AB', 'S01.0x', '0.5ABC']
for tok in tests:
    result = repair_merged_token(tok)
    print(f"  {tok!r:20s} -> {result!r}")
