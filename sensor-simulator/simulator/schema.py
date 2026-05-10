from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class TelemetryRecord:
    machine_id: str
    machine_type: str
    site_id: str
    temperature: float
    vibration: float
    rpm: int
    pressure: float
    voltage: float
    humidity: float
    timestamp: datetime
    anomaly_score: float
    degradation_index: float
    failed: int

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["timestamp"] = self.timestamp.isoformat()
        return payload
