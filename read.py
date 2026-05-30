import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

VAULT_PATH = Path(os.getenv("VAULT_PATH", "/home/prabhu/Documents/obsidian"))
OUTPUT_PATH = Path("obsidian_export.json")
HEADING_PATTERN = re.compile(r"^(#{1,2})\s+(.*\S)\s*$")


def extract_headings(text: str) -> list[str]:
    headings: list[str] = []
    for line in text.splitlines():
        match = HEADING_PATTERN.match(line)
        if match:
            headings.append(match.group(2))
    return headings


def export_markdown_vault() -> list[dict[str, str | list[str]]]:
    exported_files: list[dict[str, str | list[str]]] = []

    for file_path in VAULT_PATH.rglob("*.md"):
        try:
            text = file_path.read_text(encoding="utf-8")
        except Exception as e:
            exported_files.append(
                {
                    "filepath": str(file_path),
                    "filename": file_path.name,
                    "heading": [],
                    "last updated date&time": "",
                    "contents": f"ERROR: {e}",
                }
            )
            continue

        modified_at = datetime.fromtimestamp(
            file_path.stat().st_mtime, tz=timezone.utc
        ).astimezone()

        exported_files.append(
            {
                "filepath": str(file_path),
                "filename": file_path.name,
                "heading": extract_headings(text),
                "last updated date&time": modified_at.isoformat(timespec="seconds"),
                "contents": text,
            }
        )

    return exported_files


def main() -> None:
    exported_files = export_markdown_vault()
    OUTPUT_PATH.write_text(
        json.dumps(exported_files, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Exported {len(exported_files)} markdown files to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()