# Installation

1. Copy `cfm_agent_v_5_0` into `implementations/energy_oil_forecasting/` as a new sibling package.
2. Do not overwrite, rename, or modify any earlier CFM package.
3. From inside the package directory, verify the manifest with `sha256sum -c MANIFEST.sha256`.
4. Run the four offline checks in `README.md`.
5. Restart the notebook kernel or Python process so imports resolve to the new package.
6. Instantiate fresh `CfmV50Settings`, agent configuration, and predictor objects.
7. Run the audit-off live acceptance test before audit-on or repeated-run testing.
8. Preserve complete raw JSON outputs for review.
