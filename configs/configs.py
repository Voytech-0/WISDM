from dataclasses import dataclass


@dataclass(init=False, frozen=True)
class DataConfig:
    frequency: int = 20
    feature_length_in_seconds: int = 10

