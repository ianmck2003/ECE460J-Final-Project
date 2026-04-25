# Manual Webcam Sweep Protocol

Use this protocol to capture reproducible human trials for runtime UX settings.

## Required sweeps

- Smoothing window: `3`, `5`, `7`, `10`
- Confirmation threshold (`CONFIRM_FRAMES`): `30`, `40`, `60`

## Trial instructions

1. Pick one test word from a fixed prompt list (5-10 words).
2. Record tester identity, location, and lighting conditions.
3. Run inference with one parameter set at a time.
4. Enter predicted text and mark correctness.
5. Repeat for multiple testers and environments.

## Logging format

Append each trial to `manual_trial_log.csv` using the schema in `manual_trial_template.csv`.

Suggested starter command:

```powershell
Copy-Item analysis/manual_trials/manual_trial_template.csv analysis/manual_trials/manual_trial_log.csv
```
