"""Trusted publisher identities for package-local source resolution."""

from __future__ import annotations

from dataclasses import dataclass

from energy_oil_forecasting.cfm_agent_v_5_0.schemas import SourceTier


@dataclass(frozen=True)
class PublisherIdentity:
    """Python-owned publisher identity derived from a resolved domain."""

    publisher: str
    source_tier: SourceTier
    is_primary_or_official: bool


_TRUSTED_DOMAINS: dict[str, PublisherIdentity] = {
    "eia.gov": PublisherIdentity("U.S. Energy Information Administration", "tier_1_primary", True),
    "energy.gov": PublisherIdentity("U.S. Department of Energy", "tier_1_primary", True),
    "iea.org": PublisherIdentity("International Energy Agency", "tier_1_primary", True),
    "opec.org": PublisherIdentity("OPEC", "tier_1_primary", True),
    "api.org": PublisherIdentity("American Petroleum Institute", "tier_1_primary", True),
    "whitehouse.gov": PublisherIdentity("The White House", "tier_1_primary", True),
    "treasury.gov": PublisherIdentity("U.S. Department of the Treasury", "tier_1_primary", True),
    "state.gov": PublisherIdentity("U.S. Department of State", "tier_1_primary", True),
    "maritime.dot.gov": PublisherIdentity("U.S. Maritime Administration", "tier_1_primary", True),
    "reuters.com": PublisherIdentity("Reuters", "tier_2_independent", False),
    "bloomberg.com": PublisherIdentity("Bloomberg", "tier_2_independent", False),
    "apnews.com": PublisherIdentity("Associated Press", "tier_2_independent", False),
    "ft.com": PublisherIdentity("Financial Times", "tier_2_independent", False),
    "wsj.com": PublisherIdentity("The Wall Street Journal", "tier_2_independent", False),
    "cnbc.com": PublisherIdentity("CNBC", "tier_2_independent", False),
    "spglobal.com": PublisherIdentity("S&P Global", "tier_2_independent", False),
    "argusmedia.com": PublisherIdentity("Argus Media", "tier_2_independent", False),
    "kpler.com": PublisherIdentity("Kpler", "tier_2_independent", False),
    "rystadenergy.com": PublisherIdentity("Rystad Energy", "tier_2_independent", False),
    "lloydslist.com": PublisherIdentity("Lloyd's List", "tier_2_independent", False),
    "tradewindsnews.com": PublisherIdentity("TradeWinds", "tier_2_independent", False),
    "oilprice.com": PublisherIdentity("OilPrice.com", "tier_3_commentary", False),
}


def normalize_domain(domain: str) -> str:
    """Normalize a host for registry matching."""
    normalized = domain.strip().lower().rstrip(".")
    return normalized.removeprefix("www.")


def publisher_for_domain(domain: str) -> PublisherIdentity | None:
    """Return the most-specific trusted identity for a host or subdomain."""
    host = normalize_domain(domain)
    matches = [
        (registered, identity)
        for registered, identity in _TRUSTED_DOMAINS.items()
        if host == registered or host.endswith(f".{registered}")
    ]
    if not matches:
        return None
    return max(matches, key=lambda item: len(item[0]))[1]


__all__ = [
    "PublisherIdentity",
    "normalize_domain",
    "publisher_for_domain",
]
