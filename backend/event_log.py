"""Small JSON Lines logger for concise, non-secret search diagnostics."""

import json


def log_event(stage: str, **fields: object) -> None:
    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False, separators=(",", ":")), flush=True)
