# analyze_video prompt changelog (plan/03a section 6.1: at most two DEV iterations, each logged)

## v0 (2026-11-03)
First draft from the checklist. DEV run `20261103T233247Z_4d7bfb6`, 10 videos, 10/10 valid, $0.33.

## v1 (2026-11-03), DEV iteration 1 of 2
Misreads found in v0 on DEV:
- `v874ba35b`: last reveal at 22.6 s, video ends 28.0 s mid-animation, labelled `short_tail` (tail is 5.4 s).
  Unclear whether footage still showing the revealed thing counts as tail.
- `v952137ad`: premise text on screen at 0.0 s (`t_premise_clear_s` 0.0) but `t_first_event_s` 0.6, against the
  definition (premise text counts as the first event).

Changes:
- `payoffs.t_s`: a payoff lands at the moment of the reveal or event; footage that keeps showing it is time after
  the payoff.
- `t_first_event_s`: premise text at the start is the first event; never later than `t_premise_clear_s`.
- Validator: new quality flag `first_event_after_premise` (premise clear more than 0.5 s before the first event).

Watch item, not changed: `rewatch_trigger` was `none` on 9/10 (left to the prevalence rule, 6.2b).

DEV run `20261103T234625Z_4d7bfb6`, same 10 videos, 10/10 valid, $0.32. Both fixes held (`v874ba35b` now
`long_tail`, `v952137ad` first event 0.0). New finding: `v07f7e7da` `dead_time_s` 2.8 (v0, outro only) vs 9.0
(v1, outro plus the punchline frame held 7.8-14.2 s). That post-payoff time is already `tail_after_last_payoff_s`,
so both core features measured the same stretch.

## v2 (2026-11-03), DEV iteration 2 of 2 (last allowed)
Changes:
- `dead_time_s`: counted only between `t_first_event_s` and the last payoff; nothing after the last payoff; no
  payoffs means count to the end. "an outro" removed from the examples (it is post-payoff). Checklist C6 updated.
- Validator: new quality flag `dead_time_past_last_payoff` (dead time longer than first event to last payoff,
  plus 0.5 s).
