from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
import math
import random
from typing import Iterator

from .schema import TelemetryRecord


@dataclass(frozen=True, slots=True)
class MachineProfile:
    machine_id: str
    machine_type: str
    site_id: str
    base_temperature: float
    base_vibration: float
    base_rpm: int
    base_pressure: float
    base_voltage: float
    base_humidity: float


MACHINE_BASELINES: dict[str, dict[str, float]] = {
    "motor": {
        "temperature": 72.0,
        "vibration": 0.34,
        "rpm": 1480,
        "pressure": 28.0,
        "voltage": 415.0,
        "humidity": 41.0,
    },
    "pump": {
        "temperature": 66.0,
        "vibration": 0.28,
        "rpm": 1760,
        "pressure": 41.0,
        "voltage": 402.0,
        "humidity": 44.0,
    },
    "compressor": {
        "temperature": 81.0,
        "vibration": 0.47,
        "rpm": 3520,
        "pressure": 58.0,
        "voltage": 438.0,
        "humidity": 37.0,
    },
    "conveyor": {
        "temperature": 61.0,
        "vibration": 0.22,
        "rpm": 920,
        "pressure": 19.0,
        "voltage": 388.0,
        "humidity": 46.0,
    },
}


def build_machine_profiles(machine_count: int, seed: int = 42) -> list[MachineProfile]:
    random.seed(seed)
    machine_types = list(MACHINE_BASELINES)
    profiles: list[MachineProfile] = []
    for index in range(machine_count):
        machine_type = machine_types[index % len(machine_types)]
        baseline = MACHINE_BASELINES[machine_type]
        profiles.append(
            MachineProfile(
                machine_id=f"{machine_type.upper()}_{index + 1:03d}",
                machine_type=machine_type,
                site_id=f"SITE_{(index % 3) + 1}",
                base_temperature=baseline["temperature"] + random.uniform(-3.0, 3.0),
                base_vibration=baseline["vibration"] + random.uniform(-0.05, 0.05),
                base_rpm=int(baseline["rpm"] + random.randint(-25, 25)),
                base_pressure=baseline["pressure"] + random.uniform(-2.0, 2.0),
                base_voltage=baseline["voltage"] + random.uniform(-6.0, 6.0),
                base_humidity=baseline["humidity"] + random.uniform(-4.0, 4.0),
            )
        )
    return profiles


def generate_records(
    profiles: list[MachineProfile],
    start_time: datetime,
    periods: int,
    frequency_minutes: int = 5,
    anomaly_probability: float = 0.03,
    failure_probability: float = 0.005,
    seed: int = 42,
) -> Iterator[TelemetryRecord]:
    random.seed(seed)

    for step in range(periods):
        event_time = start_time + timedelta(minutes=step * frequency_minutes)
        day_fraction = step / max(periods, 1)

        for profile in profiles:
            periodic_load = math.sin(step / 12.0) * 2.5
            degradation = min(day_fraction * random.uniform(0.3, 1.0), 1.0)
            anomaly_triggered = random.random() < anomaly_probability
            failed = 1 if random.random() < failure_probability * (1 + degradation * 3.0) else 0

            temperature = profile.base_temperature + periodic_load + degradation * 17.0
            vibration = profile.base_vibration + degradation * 0.55
            rpm = profile.base_rpm - int(degradation * 120)
            pressure = profile.base_pressure + degradation * 7.0
            voltage = profile.base_voltage + random.uniform(-4.0, 4.0)
            humidity = profile.base_humidity + math.cos(step / 9.0) * 3.0

            if anomaly_triggered:
                temperature += random.uniform(8.0, 15.0)
                vibration += random.uniform(0.15, 0.35)
                pressure += random.uniform(2.0, 5.0)

            if failed:
                temperature += random.uniform(18.0, 28.0)
                vibration += random.uniform(0.3, 0.6)
                rpm -= random.randint(180, 300)
                pressure += random.uniform(5.0, 9.0)

            anomaly_score = round(
                0.35 * max(0.0, temperature - profile.base_temperature)
                + 32.0 * max(0.0, vibration - profile.base_vibration)
                + 0.06 * abs(rpm - profile.base_rpm)
                + 0.55 * max(0.0, pressure - profile.base_pressure),
                4,
            )

            yield TelemetryRecord(
                machine_id=profile.machine_id,
                machine_type=profile.machine_type,
                site_id=profile.site_id,
                temperature=round(temperature, 2),
                vibration=round(vibration, 4),
                rpm=max(rpm, 0),
                pressure=round(pressure, 2),
                voltage=round(voltage, 2),
                humidity=round(humidity, 2),
                timestamp=event_time,
                anomaly_score=anomaly_score,
                degradation_index=round(degradation, 4),
                failed=failed,
            )
