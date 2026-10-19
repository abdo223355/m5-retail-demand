# Checkpoints

## Current phase
**Phase: Modeling & Evaluation (ML)** — repository refactoring done; validation protocol still unresolved.

Open question of the current step:
> Unify validation on ONE time-based split and re-evaluate Day 6 and Day 7 on it.

### Phase exit checklist
- [x] Pipeline extracted to `src/`, tests passing, artifacts reproduced on samples
- [x] Documentation written (`docs/`, README)
- [ ] One shared time-based validation set for all models
- [ ] Day 6 and Day 7 re-evaluated on it (decision on the tuned model recorded)
- [ ] Forecast for the 28-day hold-out produced (no `predict` step yet)
- [ ] Known issues decided: `is_weekend`, `price_max` / back-fill
- [ ] Owner confirms: "نقفل Phase"

## Parking lot
| Parked on | Belongs to | Item |
|---|---|---|
| 2026-10-13 | Deployment / MLOps | "عايز اعمل deploy للموديل" — target (API / batch / demo), packaging, inference-time feature building |
