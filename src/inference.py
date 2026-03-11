import pickle
from pgmpy.inference import VariableElimination
from src.preprocess import extract_features

#  LOAD MODEL
def load_model(model_path):
    """Load the saved Bayesian Network model from disk."""
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    return model


#  PREDICT

def predict(text, model):
    """
    Given raw user text and a trained Bayesian Network model:
    1. Extract features from the text
    2. Run Variable Elimination inference
    3. Return probabilities and verdict
    """

    # Step 1 — Extract features from user input
    features = extract_features(text)

    # Step 2 — Run Variable Elimination
    infer = VariableElimination(model)

    result = infer.query(
        variables=['label'],
        evidence=features,
        show_progress=False
    )

    # Step 3 — Get probabilities
    p_real = round(float(result.values[0]), 2)   # index 0 = label=0 = Real
    p_fake = round(float(result.values[1]), 2)   # index 1 = label=1 = Fake

    verdict = "FAKE NEWS" if p_fake > p_real else "REAL NEWS"
    confidence = max(p_fake, p_real)

    return {
        'features'   : features,
        'p_fake'     : p_fake,
        'p_real'     : p_real,
        'verdict'    : verdict,
        'confidence' : confidence,
    }


#  DISPLAY RESULT (called from main.py)

def display_result(text, model):
    """Full pipeline: take text, predict, and print formatted result."""

    result = predict(text, model)
    features = result['features']

    print()
    print("─" * 45)
    print("  Extracted Features:")
    print(f"  • Emotional language detected   : {'YES' if features['is_emotional'] else 'NO'}")
    print(f"  • Excessive caps in title        : {'YES' if features['title_caps'] else 'NO'}")
    print(f"  • Contains numbers/statistics    : {'YES' if features['has_numbers'] else 'NO'}")
    print(f"  • Article is very short          : {'YES' if features['short_article'] else 'NO'}")
    print("─" * 45)
    print()

    # Build probability bars
    fake_bar  = "█" * int(result['p_fake'] * 20) + "░" * (20 - int(result['p_fake'] * 20))
    real_bar  = "█" * int(result['p_real'] * 20) + "░" * (20 - int(result['p_real'] * 20))

    print("         PREDICTION RESULT")
    print("─" * 45)
    print(f"  P(Fake)  =  {result['p_fake']:.2f}  {fake_bar}")
    print(f"  P(Real)  =  {result['p_real']:.2f}  {real_bar}")
    print()
    print(f"  Verdict  :  {result['verdict']}  ({int(result['confidence']*100)}% confidence)")
    print("─" * 45)
    print()