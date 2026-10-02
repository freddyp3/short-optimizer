# Brawl Stars Shorts: creator domain notes

Salvaged from the old `architecture.md` (thumbnail/CTR design, dropped 2026-09-25). These are the
creator's own hypotheses, not validated. Use them as a starting taxonomy for the `analyze_video`
prompt and as candidate features to test in evals.

## Moments that (I believe) hold viewers

- Clutch moments: low-health wins, last-second plays
- High damage numbers on screen
- Recognizable brawlers, especially meta-relevant ones
- Explosive visuals: abilities firing, team wipes
- Game-ending screens (victory / defeat)
- Unusual or funny in-game situations

## Visual signals worth testing

| Signal | Why it might matter |
|---|---|
| Damage number visible | Big numbers = hype |
| Health bar low | Clutch signal |
| Victory / defeat screen | High-engagement frames |
| Colour saturation spike | Ability animations are bright |
| Motion level | Action vs downtime |
| Brawler ability firing | Peak excitement |
| Number of brawlers visible | Team moments > solo |
| Zoomed / dramatic framing | Cinematic moments |

## Community title style

- Hook in the first 3 words; reference brawlers by name if meta-relevant
- Examples: "Surge hits different at max level 💀", "They didn't expect THIS in ranked...",
  "1v3 clutch no one talks about"
