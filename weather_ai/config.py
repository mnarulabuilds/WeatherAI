from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    """Hyperparameters tuned for speed, memory, and generalization."""

    days_per_year: int = 365
    holdout_fraction: float = 0.15
    random_state: int = 42

    reg_hidden_layers: tuple[int, ...] = (48, 24)
    reg_activation: str = "relu"
    reg_alpha: float = 0.001
    reg_max_iter: int = 400
    reg_learning_rate_init: float = 0.001

    clf_hidden_layers: tuple[int, ...] = (32, 16)
    clf_activation: str = "relu"
    clf_alpha: float = 0.001
    clf_max_iter: int = 400
    clf_learning_rate_init: float = 0.001

    use_float32: bool = True
