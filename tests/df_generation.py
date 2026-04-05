import numpy as np
import pandas as pd


def create_rf_friendly_dataset(n_samples=2000, random_state=42, noise_level=0.5):
    """
    Create a single DataFrame suitable for testing RandomForestClassifier and
    RandomForestRegressor where the 'year' column (2000-2010) is present but
    intentionally does NOT influence either target.

    Returns a DataFrame with columns:
      - X1..X6 : numeric features (mixture of continuous, binary, uniform)
      - year   : int in [2000, 2010] inclusive (no effect on y or y_binary)
      - y_binary : binary target (tree-friendly, independent of year)
      - y        : continuous target (independent of year)

    Parameters
    ----------
    n_samples : int
        Number of rows to generate.
    random_state : int
        Seed for reproducibility.
    noise_level : float
        Std dev of Gaussian noise added to the regression target.
    """
    rng = np.random.default_rng(random_state)
    n = int(n_samples)

    # Features (informative for both tasks)
    X1 = rng.normal(loc=0.5, scale=1.0, size=n)
    X2 = rng.normal(loc=-0.3, scale=1.2, size=n)
    X3 = rng.normal(loc=0.0, scale=1.0, size=n)
    X4 = rng.normal(size=n)
    X5 = rng.uniform(-2, 2, size=n)
    X6 = rng.binomial(1, 0.35, size=n)  # binary feature

    # Year column 2000..2010 inclusive (explicitly NOT used to generate targets)
    year = rng.integers(2000, 2011, size=n)

    # Classification target: rule/threshold-based (NO year effect)
    clf_score = (
        (X1 > 0.6).astype(int) * 2
        + (X2 < -0.8).astype(int) * 1
        + (X3 > 1.0).astype(int) * 1
        + (X6 == 1).astype(int) * 1
    )
    clf_continuous = 0.7 * (X4 > 0.2).astype(int) + 0.3 * (X5 > 0.5).astype(int)
    logits = 1.2 * clf_score + clf_continuous - 0.5
    probs = 1.0 / (1.0 + np.exp(-logits))
    y_binary = (rng.random(n) < probs).astype(int)

    # Regression target: linear + nonlinear + interaction (NO year effect)
    y = (
        3.0 * X1
        - 2.0 * X2
        + 4.5 * X6
        + 2.5 * (X5 * X3)  # nonlinear interaction
        + rng.normal(0, noise_level, size=n)
    )

    df = pd.DataFrame(
        {
            "X1": X1,
            "X2": X2,
            "X3": X3,
            "X4": X4,
            "X5": X5,
            "X6": X6,
            "year": year.astype(int),
            "y_binary": y_binary,
            "y": y,
        }
    )
    return df


# Example usage:
# df = create_rf_friendly_dataset(n_samples=2000, random_state=2025, noise_level=0.5)
