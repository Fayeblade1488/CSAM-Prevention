
# General AI-Agent SOP (Provider-Agnostic)

## 0) North Star
Be useful, safe, and specific. Ship small, reversible changes with tests and docs.

## 1) Intake
- Restate goal in one sentence.
- Identify constraints (security/SLAs/perf/style).
- Ask at most *one* clarifying question; if not essential, choose a sensible default.

**Template**
```
Goal: …
Constraints: …
Assumptions: …
Plan (bullets): …
Definition of Done: …
```

## 2) Plan
- 3–7 steps, tools/agents chosen, success checks defined.

## 3) Execute
- Small PRs, co-located tests, docs updated, idempotent scripts.

## 4) Verify
- Unit/integration tests, static analysis, secret scan, license check.
- Dry-run configs; show diffs.

## 5) Report
- Brief summary, diffs, next steps; structured artifacts.

## 6) Safety/Privacy/Compliance
- Never paste secrets/PII.
- Redact logs and examples.
- Escalate sensitive details privately.
- Respect licenses; add headers as needed.

## 7) Git Hygiene
- Conventional commits (`type(scope): summary`).
- Small PRs; link issues with `Fixes #123`.

## 8) Definition of Done (Universal)
- [ ] CI green (tests, lint, type, security).
- [ ] Docs & changelog updated.
- [ ] Backward compatibility considered or migration provided.
- [ ] Rollback plan noted.
- [ ] Repro steps included for reviewers.
