# Kickoff Prompt for the Next Session

Copy everything inside the block below into a new session.

```
I'm continuing work on topological-photonic-crystal-optimizer. Start by reading
docs/HANDOFF.md and CLAUDE.md. The latest work is on branch
claude/dreamy-newton-ho6psr; check whether it has been merged into main, and base
your work on whichever is newer (merge if they have diverged).

Setup: create a venv, `pip install -r requirements.txt pytest`, run `pytest`, and tell
me the actual result before changing anything (the handoff expects 68 passed).

Then show me the "Known issues" list from docs/HANDOFF.md and ask which ones to work
on. Make one atomic commit per logical change, run pytest after each, and push after
every commit. Report test results honestly, including failures.
```
