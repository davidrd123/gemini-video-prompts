"""Export the local MCP schemas as documentation, without invoking any tools."""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from gemini_video_prompts_mcp.server import mcp as generation
from media_analysis_mcp.server import mcp as analysis


ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "docs" / "TOOL_REFERENCE.md"


async def render() -> str:
    lines = [
        "# MCP tool reference", "",
        "Generated from the registered tools in this checkout. Do not edit by hand.", "",
        "Start with [the agent quickstart](AGENT_QUICKSTART.md). These are local MCP",
        "tool names, not OpenAI built-in ImageGen or provider REST endpoints. Your",
        "client may prefix names with its configured server alias. The live",
        "`tools/list` response is authoritative for the running server; restart",
        "the server if it differs from this checkout.", "",
        "Update: `uv run python scripts/update_tool_reference.py`.",
        "Check for drift: `uv run python scripts/update_tool_reference.py --check`.",
        "Neither command calls a provider, reads credentials, or generates media.", "",
        "The JSON blocks are input schemas, not example calls. `required` lists",
        "mandatory arguments. A nullable model/API default can be resolved inside",
        "the tool; see its description for the effective provider-specific default.", "",
    ]
    for command, mcp in (("gemini-prompts-mcp", generation), ("media-analysis-mcp", analysis)):
        tools = await mcp.list_tools()
        lines.extend([f"## {command}", "", f"{len(tools)} tools.", ""])
        lines.extend(f"- [{tool.name}](#{tool.name})" for tool in tools)
        lines.append("")
        for tool in tools:
            lines.extend([
                f"### {tool.name}", "", (tool.description or "").strip(), "",
                "```json", json.dumps(tool.inputSchema, indent=2, ensure_ascii=False), "```", "",
            ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Fail if the saved reference is stale.")
    args = parser.parse_args()
    expected = asyncio.run(render())
    if args.check:
        if not TARGET.is_file() or TARGET.read_text() != expected:
            print("Tool reference is stale. Run: uv run python scripts/update_tool_reference.py")
            return 1
        print("Tool reference matches the registered MCP schemas.")
        return 0
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    TARGET.write_text(expected)
    print(f"Wrote {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
