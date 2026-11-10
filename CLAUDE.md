@AGENTS.md

# Main session
- Own the working result end to end. You may edit `src/`, `tests/`, `configs/` and `scripts/`, and you must run the checks that cover your changes.
- Talk to the user in Russian; write files in English.
- Use `.claude/skills/deliver-change/SKILL.md`; there are no mandatory analyst, spec-auditor, prototype, refactor and test stages.
- Subagents are optional. Use `.claude/agents/investigator.md` for isolated diagnosis and `.claude/agents/reviewer.md` for high-risk independent review.

## Permissions
- `CLAUDE.md`, `AGENTS.md`, `.claude/agents/`: change only on the user's explicit request.
- `specs/`: new or changed contracts require the user's approval. Contract-preserving bug fixes do not.
- `.claude/skills/`: keep short and consistent with AGENTS.md; remove obsolete rules instead of accumulating process.
- Commits: only when the user asks.
