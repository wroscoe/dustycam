# Development Notes for AI/Automations

- Use standard `python` and `pip` to run tooling and tests in this repo.
- Prefer `python -m pytest ...` or `.venv/bin/pytest` instead of invoking system Python directly.
- Keep changes confined to the workspace; do not mutate files outside the project root.
- When adding dependencies, edit `pyproject.toml` or use `pip install <package>` and update requirements.
