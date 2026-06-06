from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from utils.file_utils import copy_file, load_json, write_json


@dataclass
class ErrorRecord:
    pdf: str
    fehler: str
    schicht: str
    versuche: int
    zeitpunkt: str


class ErrorQueue:
    def __init__(self, error_dir: str) -> None:
        self.error_dir = Path(error_dir)
        self.error_report = self.error_dir / "fehlerbericht.json"

    def add(self, pdf_path: str, error: Exception | str, layer: str, attempts: int) -> None:
        self.error_dir.mkdir(parents=True, exist_ok=True)
        copy_file(pdf_path, self.error_dir)
        error_records = load_json(self.error_report, default=[])
        error_records.append(
            asdict(
                ErrorRecord(
                    pdf=Path(pdf_path).name,
                    fehler=str(error),
                    schicht=layer,
                    versuche=attempts,
                    zeitpunkt=datetime.now(timezone.utc).isoformat(),
                )
            )
        )
        write_json(self.error_report, error_records)
