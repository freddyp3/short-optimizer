"""Draft quota and spending caps (plan/10a section 4, plan/10 section 12.2-12.3)."""

import threading
from datetime import datetime, timezone

import pytest

from src.server.config import Config
from src.server.quota import BudgetExceeded, LimitReached, Quota

CFG = Config(DRAFTS_PER_DAY=3, OWN_KEY_DRAFTS_PER_DAY=4, DAY_CAP_USD=1.00, MONTH_CAP_USD=10.00,
             EST_COST_PER_DRAFT_USD=0.12)


class Clock:
    def __init__(self, *ymdhm: int):
        self.t = datetime(*ymdhm, tzinfo=timezone.utc).timestamp()

    def __call__(self) -> float:
        return self.t


@pytest.fixture
def clock():
    return Clock(2026, 11, 14, 22, 30)


@pytest.fixture
def quota(tmp_path, clock):
    return Quota(tmp_path / "server.sqlite3", clock, CFG)


def _refused(quota: Quota, token_id: str, draft_id: str, paid_by: str = "server") -> LimitReached:
    with pytest.raises(LimitReached) as err:
        quota.check_and_reserve(token_id, draft_id, paid_by)
    return err.value


def test_reservation_counts_as_a_draft_and_as_spend(quota):
    assert quota.usage("tok-a") == {"drafts_today": 0, "drafts_left_today": 3, "drafts_per_day": 3,
                                    "own_key_drafts_today": 0, "own_key_drafts_left_today": 4,
                                    "resets_at_utc": "2026-11-15T00:00:00Z"}
    quota.check_and_reserve("tok-a", "d1", "server")
    usage = quota.usage("tok-a")
    assert (usage["drafts_today"], usage["drafts_left_today"]) == (1, 2)
    assert quota.usage("tok-b")["drafts_today"] == 0
    summary = quota.summary()
    assert (summary["day"], summary["drafts"], summary["spend_usd"], summary["month_spend_usd"]) == \
        ("2026-11-14", 1, 0.12, 0.12)


def test_user_daily_limit(quota, clock):
    for i in range(3):
        quota.check_and_reserve("tok-a", f"d{i}", "server")
    refusal = _refused(quota, "tok-a", "d3")
    assert (refusal.kind, refusal.retry_after_utc) == ("user_daily", "2026-11-15T00:00:00Z")
    quota.check_and_reserve("tok-b", "other", "server")
    assert quota.usage("tok-a")["drafts_today"] == 3, "a refused attempt is not a draft"
    assert quota.summary()["refusals"]["user_daily"] == 1 and quota.summary()["drafts"] == 4

    clock.t += 2 * 3600
    quota.check_and_reserve("tok-a", "d3", "server")
    assert quota.usage("tok-a") == {**quota.usage("tok-a"), "drafts_today": 1, "resets_at_utc": "2026-11-16T00:00:00Z"}


def test_settled_drafts_still_count_toward_the_daily_limit(quota):
    for i in range(3):
        quota.check_and_reserve("tok-a", f"d{i}", "server")
        quota.settle(f"d{i}", 0.0)
    assert _refused(quota, "tok-a", "d3").kind == "user_daily"


def test_day_cap(quota, clock):
    for i in range(8):
        quota.check_and_reserve(f"tok-{i}", f"d{i}", "server")
    assert quota.summary()["spend_usd"] == 0.96
    refusal = _refused(quota, "tok-new", "d8")
    assert (refusal.kind, refusal.retry_after_utc) == ("day_cap", "2026-11-15T00:00:00Z")
    assert quota.usage("tok-new")["drafts_today"] == 0
    clock.t += 2 * 3600
    quota.check_and_reserve("tok-new", "d8", "server")


def test_day_cap_is_reachable_exactly(tmp_path, clock):
    quota = Quota(tmp_path / "q.sqlite3", clock, Config(DAY_CAP_USD=0.3, EST_COST_PER_DRAFT_USD=0.1))
    for i in range(3):
        quota.check_and_reserve(f"tok-{i}", f"d{i}", "server")
    assert _refused(quota, "tok-x", "dx").kind == "day_cap"


def test_day_cap_uses_actual_cost_after_settling(quota):
    for i in range(8):
        quota.check_and_reserve(f"tok-{i}", f"d{i}", "server")
    assert _refused(quota, "tok-new", "x1").kind == "day_cap"
    quota.settle("d0", 0.02)
    quota.check_and_reserve("tok-new", "x2", "server")
    quota.settle("d1", 0.30)
    assert _refused(quota, "tok-other", "x3").kind == "day_cap"


def test_running_cost_above_the_estimate_counts_before_settling(quota):
    for i in range(7):
        quota.check_and_reserve(f"tok-{i}", f"d{i}", "server")
    quota.add_cost("d0", 0.05)
    assert quota.summary()["spend_usd"] == 0.84, "below the estimate the reservation stands"
    quota.add_cost("d0", 0.17)
    assert quota.summary()["spend_usd"] == 0.94
    assert _refused(quota, "tok-new", "x").kind == "day_cap"


def test_settled_spend_above_the_estimate_counts_against_the_caps(quota):
    for i in range(3):
        quota.check_and_reserve(f"tok-{i}", f"d{i}", "server")
        quota.add_cost(f"d{i}", 0.24)                       # 2 x the estimate: the most a job may record without being stopped
        quota.settle(f"d{i}", 0.24)
    assert quota.summary()["spend_usd"] == 0.72 == quota.summary()["settled_usd"]
    quota.check_and_reserve("tok-3", "d3", "server")        # 0.72 + 0.12 = 0.84
    quota.check_and_reserve("tok-4", "d4", "server")        # 0.84 + 0.12 = 0.96
    refusal = _refused(quota, "tok-5", "d5")                # 0.96 + 0.12 > 1.00; at the estimate (0.60) it would have passed
    assert (refusal.kind, refusal.retry_after_utc) == ("day_cap", "2026-11-15T00:00:00Z")
    quota.add_cost("d3", 0.24)                              # an open draft counts at max(reservation, running cost)
    quota.cancel("d4")
    assert quota.summary()["spend_usd"] == 0.96             # 0.72 settled + 0.24 running; the cancelled draft is gone
    assert _refused(quota, "tok-5", "d5").kind == "day_cap"
    assert quota.summary()["drafts"] == 4 and quota.summary()["refusals"]["day_cap"] == 2


def test_month_cap_counts_settled_spend_above_the_estimate(tmp_path, clock):
    cfg = Config(DRAFTS_PER_DAY=3, DAY_CAP_USD=1.0, MONTH_CAP_USD=0.50, EST_COST_PER_DRAFT_USD=0.12)
    quota = Quota(tmp_path / "q.sqlite3", clock, cfg)
    for i in range(2):
        quota.check_and_reserve("tok-a", f"d{i}", "server")
        quota.settle(f"d{i}", 0.24)
        clock.t += 86400
    assert _refused(quota, "tok-a", "d2").kind == "month_cap"      # 0.48 + 0.12 > 0.50


def test_month_cap(tmp_path, clock):
    cfg = Config(DRAFTS_PER_DAY=3, DAY_CAP_USD=1.0, MONTH_CAP_USD=0.30, EST_COST_PER_DRAFT_USD=0.12)
    quota = Quota(tmp_path / "q.sqlite3", clock, cfg)
    quota.check_and_reserve("tok-a", "d0", "server")
    clock.t += 86400
    quota.check_and_reserve("tok-a", "d1", "server")
    clock.t += 86400
    refusal = _refused(quota, "tok-a", "d2")
    assert (refusal.kind, refusal.retry_after_utc) == ("month_cap", "2026-12-01T00:00:00Z")
    assert quota.summary()["refusals"] == {"user_daily": 0, "day_cap": 0, "month_cap": 1, "own_key_daily": 0}
    assert quota.summary()["month_spend_usd"] == 0.24 and quota.summary()["spend_usd"] == 0
    clock.t = Clock(2026, 12, 1, 0, 0).t
    quota.check_and_reserve("tok-a", "d2", "server")


def test_month_rollover_in_december(tmp_path):
    clock = Clock(2026, 12, 31, 23, 59)
    quota = Quota(tmp_path / "q.sqlite3", clock, Config(MONTH_CAP_USD=0.12, EST_COST_PER_DRAFT_USD=0.12))
    quota.check_and_reserve("tok-a", "d0", "server")
    assert _refused(quota, "tok-b", "d1").retry_after_utc == "2027-01-01T00:00:00Z"


def test_own_key_path(quota):
    for i in range(3):
        quota.check_and_reserve("tok-a", f"s{i}", "server")
    for i in range(4):
        quota.check_and_reserve("tok-a", f"o{i}", "own_key")
    refusal = _refused(quota, "tok-a", "o4", "own_key")
    assert (refusal.kind, refusal.retry_after_utc) == ("own_key_daily", "2026-11-15T00:00:00Z")
    usage = quota.usage("tok-a")
    assert (usage["drafts_today"], usage["own_key_drafts_today"], usage["own_key_drafts_left_today"]) == (3, 4, 0)
    summary = quota.summary()
    assert (summary["drafts"], summary["own_key_drafts"], summary["spend_usd"]) == (3, 4, 0.36)
    assert summary["refusals"]["own_key_daily"] == 1


def test_own_key_ignores_the_server_caps_and_reserves_nothing(tmp_path, clock):
    cfg = Config(DRAFTS_PER_DAY=1, DAY_CAP_USD=0.12, MONTH_CAP_USD=0.12, EST_COST_PER_DRAFT_USD=0.12)
    quota = Quota(tmp_path / "q.sqlite3", clock, cfg)
    quota.check_and_reserve("tok-a", "s0", "server")
    assert _refused(quota, "tok-a", "s1").kind == "user_daily"
    assert _refused(quota, "tok-b", "s2").kind == "day_cap"
    quota.check_and_reserve("tok-a", "o0", "own_key")
    quota.add_cost("o0", 0.2)
    quota.settle("o0", 0.2)
    summary = quota.summary()
    assert (summary["spend_usd"], summary["month_spend_usd"], summary["own_key_usd"]) == (0.12, 0.12, 0.2)
    assert quota.usage("tok-a")["drafts_today"] == 1


def test_settle_on_success_replaces_the_reservation(quota):
    quota.check_and_reserve("tok-a", "d1", "server")
    quota.add_cost("d1", 0.04)
    quota.add_cost("d1", 0.05)
    quota.settle("d1", 0.09)
    summary = quota.summary()
    assert (summary["spend_usd"], summary["settled_usd"], summary["drafts"]) == (0.09, 0.09, 1)
    assert quota.usage("tok-a")["drafts_today"] == 1


def test_settle_on_failure_keeps_only_the_cost_so_far(quota):
    quota.check_and_reserve("tok-a", "d1", "server")
    quota.add_cost("d1", 0.03)
    quota.settle("d1", 0.03)
    assert quota.summary()["spend_usd"] == 0.03
    quota.check_and_reserve("tok-a", "d2", "server")
    quota.settle("d2", 0.0)
    assert quota.summary()["spend_usd"] == 0.03 and quota.usage("tok-a")["drafts_today"] == 2


def test_settle_is_idempotent_and_ignores_unknown_drafts(quota):
    quota.check_and_reserve("tok-a", "d1", "server")
    quota.settle("d1", 0.10)
    quota.settle("d1", 0.10)
    quota.settle("never-reserved", 5.0)
    assert quota.summary()["spend_usd"] == 0.10 and quota.summary()["drafts"] == 1
    with pytest.raises(ValueError):
        quota.settle("d1", -1)


def test_draft_keeps_the_day_it_was_reserved(quota, clock):
    quota.check_and_reserve("tok-a", "d1", "server")
    clock.t += 2 * 3600
    quota.add_cost("d1", 0.2)
    quota.settle("d1", 0.2)
    assert quota.summary("2026-11-14")["spend_usd"] == 0.2
    assert quota.summary()["spend_usd"] == 0 and quota.summary()["day"] == "2026-11-15"


def test_add_cost_stops_a_runaway_draft(quota):
    quota.check_and_reserve("tok-a", "d1", "server")
    quota.add_cost("d1", 0.24)                              # exactly 2 x the estimate is still allowed
    with pytest.raises(BudgetExceeded, match="passed 2x the estimate"):
        quota.add_cost("d1", 0.01)
    assert quota.summary()["spend_usd"] == 0.25, "money already spent is recorded even when the job is stopped"
    quota.settle("d1", 0.25)
    assert quota.summary()["settled_usd"] == 0.25


def test_budget_factor_is_a_setting(tmp_path, clock):
    from src.server import config

    assert Config().BUDGET_FACTOR == 2.0 and config.load({"SO_BUDGET_FACTOR": "1.5"}).BUDGET_FACTOR == 1.5
    with pytest.raises(config.ConfigError):
        Config(BUDGET_FACTOR=0.5)
    quota = Quota(tmp_path / "q.sqlite3", clock, Config(EST_COST_PER_DRAFT_USD=0.10, BUDGET_FACTOR=3.0))
    quota.check_and_reserve("tok-a", "d1", "server")
    quota.add_cost("d1", 0.30)
    with pytest.raises(BudgetExceeded):
        quota.add_cost("d1", 0.01)
    quota.check_and_reserve("tok-a", "o1", "own_key")       # an own-key job is stopped at the same point
    with pytest.raises(BudgetExceeded):
        quota.add_cost("o1", 0.31)


def test_add_cost_needs_an_open_reservation(quota):
    with pytest.raises(BudgetExceeded):
        quota.add_cost("unknown", 0.01)
    quota.check_and_reserve("tok-a", "d1", "server")
    quota.settle("d1", 0.05)
    with pytest.raises(BudgetExceeded):
        quota.add_cost("d1", 0.01)
    quota.check_and_reserve("tok-a", "o1", "own_key")
    quota.settle("o1", 0.05)
    with pytest.raises(BudgetExceeded):
        quota.add_cost("o1", 0.01)
    for bad in (-0.01, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            quota.add_cost("d1", bad)
    assert quota.summary()["spend_usd"] == 0.05


def test_refused_draft_has_no_reservation(quota):
    for i in range(3):
        quota.check_and_reserve("tok-a", f"d{i}", "server")
    _refused(quota, "tok-a", "d3")
    with pytest.raises(BudgetExceeded):
        quota.add_cost("d3", 0.01)
    quota.settle("d3", 1.0)
    assert quota.summary()["spend_usd"] == 0.36


def test_a_draft_is_reserved_once(quota):
    quota.check_and_reserve("tok-a", "d1", "server")
    with pytest.raises(ValueError):
        quota.check_and_reserve("tok-a", "d1", "server")
    with pytest.raises(ValueError):
        quota.check_and_reserve("tok-a", "d1", "own_key")
    with pytest.raises(ValueError):
        quota.check_and_reserve("tok-a", "d2", "fredrik")
    assert quota.usage("tok-a")["drafts_today"] == 1


def test_cancel_frees_an_unused_reservation(quota):
    quota.check_and_reserve("tok-a", "d1", "server")
    assert quota.cancel("d1") is True and quota.cancel("d1") is False
    assert quota.usage("tok-a")["drafts_today"] == 0 and quota.summary()["spend_usd"] == 0
    quota.check_and_reserve("tok-a", "d1", "server")
    quota.add_cost("d1", 0.01)
    assert quota.cancel("d1") is False
    quota.check_and_reserve("tok-a", "o1", "own_key")
    assert quota.cancel("o1") is True and quota.usage("tok-a")["own_key_drafts_today"] == 0


def test_reservations_are_atomic_under_threads(quota):
    barrier, outcomes = threading.Barrier(8), []

    def reserve(i: int):
        barrier.wait()
        try:
            quota.check_and_reserve("tok-a", f"d{i}", "server")
            outcomes.append("ok")
        except LimitReached as refusal:
            outcomes.append(refusal.kind)

    threads = [threading.Thread(target=reserve, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sorted(outcomes) == ["ok"] * 3 + ["user_daily"] * 5
    assert quota.usage("tok-a")["drafts_today"] == 3 and quota.summary()["refusals"]["user_daily"] == 5


def test_day_cap_is_atomic_under_threads(quota):
    barrier, outcomes = threading.Barrier(12), []

    def reserve(i: int):
        barrier.wait()
        try:
            quota.check_and_reserve(f"tok-{i}", f"d{i}", "server")
            outcomes.append("ok")
        except LimitReached as refusal:
            outcomes.append(refusal.kind)

    threads = [threading.Thread(target=reserve, args=(i,)) for i in range(12)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert sorted(outcomes) == ["day_cap"] * 4 + ["ok"] * 8
    assert quota.summary()["spend_usd"] == 0.96


def test_summary_for_a_named_day(quota, clock):
    quota.check_and_reserve("tok-a", "d1", "server")
    clock.t += 86400
    assert quota.summary("2026-11-14")["drafts"] == 1 and quota.summary()["drafts"] == 0
    assert quota.summary()["month_spend_usd"] == 0.12
    with pytest.raises(ValueError):
        quota.summary("14/11/2026")
