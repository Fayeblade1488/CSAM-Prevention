
# Agent Persona Pack (Generalized)

This pack contains ready-to-paste **agent personas**, a **Jules (orchestrator) spec**, and a **general AI Agent SOP**. 
Everything is provider-agnostic and repo-friendly.

## Contents
- `personas/` — GitHub-friendly bot personas with system prompts, capabilities, checklists.
- `jules/` — Orchestrator prompt/spec and decision policy.
- `sop/` — Universal agent Standard Operating Procedure.
- `templates/github-actions/` — Minimal, generalized GitHub Actions skeletons wired for an LLM gateway.
- `runner/agent_runner_example.json` — Provider-agnostic request shape for executing any persona consistently.

## Using This Pack
1. Paste the **System Prompt** for a persona into your agent/gateway config or secret.
2. Drop a matching **GitHub Action** skeleton into `.github/workflows/` and point it at your gateway.
3. Keep prompts in code or secrets; never hardcode tokens.
4. Let **Jules** plan work and hand tasks off to worker personas.

> Everything here is generalized by design. Replace `YOUR_*` placeholders with your org’s values.
