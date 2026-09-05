# Working in riff-mcp

## Start here

For a new session, read [the agent quickstart](docs/AGENT_QUICKSTART.md).
It covers setup, provider selection, approval, generation, edits, result paths,
and recovery without earlier chat context or private skills.
Use [the generated tool reference](docs/TOOL_REFERENCE.md) for exact input
schemas. The running MCP server's tool list must match the intended checkout;
restart it after an update before using new arguments.

## OpenAI and fal API billing

OpenAI image generation and fal H3 Max video generation use separately billed API access.
These calls do not use the user's ChatGPT/Codex subscription allowance.

- An installed `OPENAI_API_KEY` or `FAL_KEY`, a provider choice, or an auto-approved MCP
  tool is not authorization to spend money.
- Before an OpenAI generation/edit or fal video submission, tell the user that API billing applies
  and obtain approval for the specific request or a bounded batch (including
  output count and quality/size; for fal, endpoint, duration, resolution, and references). Honor existing explicit approval within that
  scope; do not repeatedly ask for the same approved work.
- Only then pass `allow_api_billing=true` to `generate_image` or `start_fal_video_job`, or
  `--allow-api-billing` to the CLI. Never set these merely to get past an error.
- Use `dry_run=true` / `--plan` to inspect requests for free. Surface the returned
  billing notice before requesting approval.
- Never silently fall back from subscription generation or another provider
  to the OpenAI or fal API. Do not retry an uncertain timeout without checking the
  saved record and obtaining authorization for another billable attempt.
- Keep keys in the user's untracked `.env` or process environment. Never print,
  commit, or copy a key into examples or another teammate's configuration.

The confirmation argument records the caller's acknowledgement; it is not a
provider-enforced spending cap or independent proof of human approval.

## Compatibility and verification

Gemini remains the default provider. Preserve legacy Gemini job hashes, output
paths, and stateful interaction behavior. Keep OpenAI imports optional and
OpenAI parameters separate from Gemini configuration.

Run `uv run --extra openai pytest -q` and `uv run --extra openai ruff check src tests`
for the full offline suite, including real-SDK tests with mocked HTTP transport.
`uv run pytest -q` verifies the base installation, with optional SDK tests skipped
when the SDK is absent. Paid live calls require the billing approval above.

After changing MCP signatures or docstrings, run
`uv run python scripts/update_tool_reference.py`, then verify with `--check`.
These commands export registered schemas without calling providers. Keep
the quickstart, README, example files, and source distribution consistent.

README.md is the current installation and usage reference. MCP_DESIGN.md retains
historical design stages; LIVE_VERIFICATION.md distinguishes live evidence from
local verification. Update relevant examples and docs when changing the tool.

For fal, read [the H3 Max guide](docs/FAL_H3_MAX.md). `start_fal_video_job`
is a separate submission tool; existing Seedance and CLI video behavior stays
unchanged. Poll and collect existing jobs with `get_video_job`; never submit
another job to recover from a timeout. Preserve both original and expanded prompts.
