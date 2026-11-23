
import numpy as np
import pandas as pd


def generate_reference_data():

    rng = np.random.default_rng(42)

    data = pd.DataFrame({
        "age": rng.normal(35, 8, 1000),
        "income": rng.normal(50000, 10000, 1000),
        "credit_score": rng.normal(700, 50, 1000)
    })

    return data


def generate_live_data(drift=False, batch_size=50):

    rng = np.random.default_rng()

    if drift:

        # Simulate a change in customer population
        age_mean = 45
        income_mean = 80000
        credit_mean = 600

    else:

        # Normal incoming customer population
        age_mean = 35
        income_mean = 50000
        credit_mean = 700

    data = pd.DataFrame({
        "age": rng.normal(
            age_mean, 8, batch_size
        ),
        "income": rng.normal(
            income_mean, 10000, batch_size
        ),
        "credit_score": rng.normal(
            credit_mean, 50, batch_size
        )
    })

    return data
