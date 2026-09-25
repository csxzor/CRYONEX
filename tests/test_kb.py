"""Offline knowledge base: every technique and countermeasure we show is real and resolvable."""

from kcwm.kb.knowledge import (
    CURATED_FALLBACK,
    _d3fend_catalogue,
    all_stage_cards,
    attack,
    countermeasures,
)
from kcwm.stages import STAGES


def test_every_stage_has_a_card_with_real_techniques():
    cards = all_stage_cards()
    assert set(cards) == set(STAGES[1:])
    for card in cards.values():
        assert card["tactic_id"].startswith("TA")
        for t in card["techniques"]:
            assert t["id"] in attack()["techniques"], t["id"]
            assert t["name"] and t["url"].startswith("https://attack.mitre.org/")


def test_curated_fallbacks_only_use_real_d3fend_ids():
    cat = _d3fend_catalogue()
    for tid, ids in CURATED_FALLBACK.items():
        for d in ids:
            assert d in cat, f"{d} (fallback for {tid}) is not a D3FEND technique we fetched"


def test_countermeasures_ranked_and_diverse():
    cms = countermeasures("T1110")
    assert 1 <= len(cms) <= 4
    assert len({c["tactic"] for c in cms[:2]}) == 2
    assert all(not c["curated"] for c in cms)
    assert all(c["curated"] for c in countermeasures("T1595"))
