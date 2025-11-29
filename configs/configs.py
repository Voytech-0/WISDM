from dataclasses import dataclass
import os

@dataclass(init=False, frozen=True)
class PathsConfig:
    z_score_params_path: str = "./data/processed/z_score_params/.csv"

@dataclass(init=False, frozen=True)
class GeneralConfig:
    seed: int = 25
    pass

@dataclass(init=False, frozen=True)
class ModelConfig:
    pass

@dataclass(init=False, frozen=True)
class DataConfig:
    frequency: int = 20
    feature_length_in_seconds: int = 10
    data_path_prefix: str = "/Users/wojtek/Library/Mobile Documents/com~apple~CloudDocs/Documents/EPRS/WISDM/data/wisdm-dataset/raw"
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1

@dataclass(init=False, frozen=True)
class MySQLConfig:
    # Read connection details from environment so Docker Compose can override them.
    host: str = os.getenv('MYSQL_HOST', 'localhost')
    port: int = int(os.getenv('MYSQL_PORT', '3306'))
    user: str = os.getenv('MYSQL_USER', 'user')
    password: str = os.getenv('MYSQL_PASSWORD', 'password')
    db: str = os.getenv('MYSQL_DATABASE', 'wisdm')