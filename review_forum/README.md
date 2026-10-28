# Review Forum — Multi-Agent Meticulous Review

10 reviewers + 1 Moderator. Read-only Round 1, debate Round 2, synthesis last.

## Pipeline

[Round 1: Solo Audits] -> 10 agents x 3-4 findings -> `round1/<role>.md`
[Clustering] -> Moderator dedupes to 8-12 canonical topics -> `topics/TOPIC-*.md`
[Round 2: Forum] -> 2 cross-role turns per thread (Debate & Veto)
[Synthesis] -> `synthesis/REPORT.md`: ranked table, doc-vs-tree drifts, pub blockers, action tickets

## Post types

- `[GRIEVANCE]` broken / misleading
- `[QUESTION]` needs evidence
- `[SUGGESTION]` concrete minimal change
- `[REBUTTAL]` disagree with file:line reason
- `[RESOLUTION]` accept / reject / needs-measurement + owner + KI link

## Rules (enforced by Moderator)

1. Every claim cites file:line or measurement + layout (`segment_samples`).
2. No `+1`s, no opinion without evidence.
3. Ledger-Guardian can one-line veto retired ideas: `REJECT — already tried, see STRATEGY_LEDGER §X`.
4. Physics/V&V can veto on safety; Student/Coach can veto on jargon.
5. Stalemate after 2 turns = `needs-measurement`, never majority vote on physics.
6. Tree wins over docs; drift becomes a ticket, not a silent fix.
