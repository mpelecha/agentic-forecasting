"""Find news sources whose URL carries a date on or after the forecast origin.

v5.2 runs keep the sources the agent was given in each prediction's
``active_research_packet.sources``. Many URLs embed their publication date
(``/2025/10/14/``, ``...-june-24-2025``, StatCan's ``daily-quotidien/YYMMDD``). This
scan compares those dates with the origin, without fetching any page. It is a lower
bound on post-origin sources: most URLs carry no date.

Usage (repo root)::

    uv run python implementations/energy_oil_forecasting/scripts/scan_source_url_dates.py \
        energy_oil_backtest_biweekly energy_oil_eval_biweekly energy_oil_eval_weekly
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
import yaml


STORE = Path(__file__).resolve().parents[1] / "data" / "predictions"
_MONTHS = "january|february|march|april|may|june|july|august|september|october|november|december"
_MONTH_NUM = {m: i + 1 for i, m in enumerate(_MONTHS.split("|"))}


def url_dates(url: str) -> list[pd.Timestamp]:
    """Return the dates written in ``url``."""
    u = url.lower()
    found = [pd.Timestamp(int(y), int(m), int(d)) for y, m, d in re.findall(r"(20\d\d)[/-](\d\d)[/-](\d\d)", u)]
    found += [
        pd.Timestamp(int(y), _MONTH_NUM[m], int(d)) for m, d, y in re.findall(rf"({_MONTHS})-(\d{{1,2}})-(20\d\d)", u)
    ]
    found += [
        pd.Timestamp(2000 + int(s[:2]), int(s[2:4]), int(s[4:])) for s in re.findall(r"daily-quotidien/(\d{6})/", u)
    ]
    return found


def scan(spec_ids: list[str]) -> pd.DataFrame:
    rows = []
    for spec_id in spec_ids:
        for path in sorted((STORE / spec_id).rglob("agent_predictor_cfm_agent_v_5_2*.yaml")):
            with open(path) as f:
                raw = yaml.safe_load(f)
            seen = set()
            for pred in raw["predictions"]:
                origin = pd.Timestamp(pred["as_of"]).normalize()
                if origin in seen:  # every horizon shares one research packet
                    continue
                seen.add(origin)
                for src in (pred["metadata"].get("active_research_packet") or {}).get("sources", []):
                    url = src.get("resolved_url") or ""
                    for date in url_dates(url):
                        rows.append(
                            {
                                "spec_id": spec_id,
                                "predictor": path.name.split("__")[0],
                                "origin": origin,
                                "url_date": date,
                                "days_after_origin": (date - origin).days,
                                "url": url,
                            }
                        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("spec_ids", nargs="+")
    args = parser.parse_args()
    df = scan(args.spec_ids)
    if df.empty:
        print("No dated URLs found.")
        return
    bins = [-(10**6), -1, 0, 7, 90, 10**6]
    labels = ["before", "same day", "1-7 days after", "8-90 days after", ">90 days after"]
    df["when"] = pd.cut(df["days_after_origin"], bins, labels=labels)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_colwidth", 110)
    print("Sources with a date in the URL, by date relative to the origin:")
    print(df.groupby("spec_id")["when"].value_counts().unstack(fill_value=0).to_string())
    late = df[df["days_after_origin"] > 0].drop_duplicates(["spec_id", "origin", "url"])
    print(f"\nDated after the origin ({len(late)}):")
    print(late.sort_values(["spec_id", "origin"])[["spec_id", "origin", "url_date", "url"]].to_string(index=False))
    out = Path(__file__).resolve().parents[1] / "results" / "v52_source_url_dates.csv"
    out.parent.mkdir(exist_ok=True)
    df.to_csv(out, index=False, date_format="%Y-%m-%d")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
