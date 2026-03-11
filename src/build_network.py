from pyexpat import model

import pandas as pd
import pickle
import os

from pgmpy.models import DiscreteBayesianNetwork as BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator

#  BAYESIAN NETWORK STRUCTURE
#
#   is_emotional ──────┐
#   title_caps ────────┼──► label (0=Real, 1=Fake)
#   has_numbers ───────┤
#   short_article ─────┘
#
# Each feature node is a PARENT of label.
# This means: the label (fake/real) depends on all 4 features.
# ─────────────────────────────────────────────

NETWORK_EDGES = [
    ('is_emotional',  'label'),
    ('title_caps',    'label'),
    ('has_numbers',   'label'),
    ('short_article', 'label'),
]

def build_and_train(features_path, model_save_path):
    """
    Load processed features, define Bayesian Network structure,
    train CPT tables using Maximum Likelihood Estimation,
    and save the trained model.
    """
    print("Loading processed features")
    df = pd.read_csv(features_path)

    # Make sure all values are integers (0 or 1)
    df = df.astype(int)

    print(f"Training on {len(df)} articles")

    # Define the network structure
    model = BayesianNetwork(NETWORK_EDGES)

    # Learn CPT tables from data
    # MaximumLikelihoodEstimator counts frequencies in data
    # to fill in Conditional Probability Tables
    model.fit(
        data=df,
        estimator=MaximumLikelihoodEstimator
    )

    # Verify the model is valid
    assert model.check_model(), "Model structure is invalid!"

    # Save the trained model
    os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
    with open(model_save_path, 'wb') as f:
        pickle.dump(model, f)

    print(f"Model trained and saved to: {model_save_path}")
    print()

    # Print CPT for label
    print("─── CPT Table for label (Fake/Real) ───")
    print(model.get_cpds('label'))
    print()

    return model


if __name__ == "__main__":
    build_and_train(
        features_path="data/processed/features.csv",
        model_save_path="models/bayesian_model.pkl"
    )