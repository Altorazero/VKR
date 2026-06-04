from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def export_json(payload: dict[str, Any], path: str) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return str(output)


def export_csv(records: list[dict[str, Any]], path: str) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(output, index=False)
    return str(output)


def export_pdf_text_report(summary: dict[str, Any], path: str) -> str:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Speech Tempo Report", "==================", ""]
    lines.extend([f"{k}: {v}" for k, v in summary.items()])
    output.write_text("\n".join(lines), encoding="utf-8")
    return str(output)
