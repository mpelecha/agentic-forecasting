---
name: research
description: Use verified cutoff-aware web evidence and expose material sources.
---

# Verified Research

1. Pass the exact task cutoff to `search_web`.
2. Use only verifier-approved output. Treat `[SEARCH_VERIFICATION_FAILED]` as no
   evidence and do not reconstruct rejected claims from memory.
3. Prefer primary sources and reputable independent reporting.
4. For every material claim, record title, URL, concise claim, and forecast
   effect in `verified_evidence`.
5. Reference evidence by zero-based index from each affected horizon.
6. Avoid double-counting facts already reflected in market data.
7. Leave evidence empty when verified research does not justify a forecast effect.
