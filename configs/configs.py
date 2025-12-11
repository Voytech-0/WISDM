from dataclasses import dataclass
import os
from pathlib import Path
from dotenv import load_dotenv
import socket

project_root = Path(__file__).resolve().parents[1]
dotenv_path = project_root / ".env"
load_dotenv(dotenv_path)


def _resolve_mysql_host():
    """Return the configured MYSQL_HOST if it resolves on this machine, otherwise fall back to 'localhost'.

    This keeps the default behavior of using 'mysql' inside Docker Compose networks
    while making host-based runs (macOS) work without editing .env.
    """
    env_host = os.getenv('MYSQL_HOST')
    if not env_host:
        return 'localhost'
    try:
        # Attempt to resolve the hostname; if it fails, fall back to localhost
        socket.getaddrinfo(env_host, None)
        return env_host
    except Exception:
        # resolution failed — fall back to localhost (useful for running on the host)
        return 'localhost'

@dataclass(init=False, frozen=True)
class PathsConfig:
    z_score_params_path: str = str(project_root / 'models' / 'processed' / 'z_score_params.csv')

@dataclass(init=False, frozen=True)
class GeneralConfig:
    seed: int = 25
    pass

@dataclass(init=False, frozen=True)
class ModelConfig:
    pass

@dataclass(init=False, frozen=True)
class DataConfig:
    db_read_chunk_size: int = 100000
    frequency: int = 20
    feature_length_in_seconds: int = 10
    data_path_prefix: str = "data/wisdm-dataset/raw"
    possible_devices: list[str] = ("phone", "watch")
    possible_sensors: list[str] = ("accel", "gyro")
    possible_labels: list[str] = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S")
    train_split: float = 0.8
    validation_split: float = 0.1
    test_split: float = 0.1

@dataclass(init=False, frozen=True)
class MySQLConfig:
    # Read connection details from environment so Docker Compose can override them.
    host: str = _resolve_mysql_host()
    port: int = int(os.getenv('MYSQL_PORT', '3306'))
    user: str = os.getenv('MYSQL_USER', 'user')
    password: str = os.getenv('MYSQL_PASSWORD', 'password')
    db: str = os.getenv('MYSQL_DATABASE', 'wisdm')

# Add training/runtime config used by src/main_training.py
@dataclass
class TrainingConfig:
    seed: int = int(os.getenv('SEED', '44'))
    max_epochs: int = int(os.getenv('MAX_EPOCHS', '10'))
    batch_size: int = int(os.getenv('BATCH_SIZE', '64'))
    lr: float = float(os.getenv('LR', '0.001'))
    hidden_size: int = int(os.getenv('HIDDEN_SIZE', '64'))
    num_layers: int = int(os.getenv('NUM_LAYERS', '1'))
    num_classes: int = int(os.getenv('NUM_CLASSES', '18'))
    # model directory and default checkpoint path
    model_dir: str = os.getenv('MODEL_DIR', str(project_root / 'models'))
    model_path: str = os.getenv('MODEL_PATH', str(project_root / 'models' / 'mlp.ckpt'))
