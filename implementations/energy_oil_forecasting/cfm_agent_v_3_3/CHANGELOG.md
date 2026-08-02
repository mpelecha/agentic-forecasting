## CFM Agent v3.3 changes from v3.2

- Replaced the single binary evidence threshold with Python-assigned limited, corroborated, and strong evidence tiers.
- Added confidence thresholds of 0.50, 0.65, and 0.80.
- Added tier-specific action caps and a 50% discount for possibly partly reflected evidence.
- Permitted explicitly verified source identity metadata with imperfect URLs, with audit warnings.
- Separated LLM-proposed, sanitized, and Python-approved assessments and actions.
- Neutralized rejected actions and removed rejected citations from approved output.
- Added source-level and claim-level evidence diagnostics.
- Retained the unadjusted authoritative ensemble as a control.
- Preserved all numerical-model and one-execution architecture from v3.2.
