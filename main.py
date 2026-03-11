import os
import sys

# Add project root to path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.preprocess    import load_and_process
from src.build_network import build_and_train
from src.inference     import load_model, display_result
from src.show_network  import show_network

FAKE_CSV      = "data/raw/Fake.csv"
REAL_CSV      = "data/raw/True.csv"
FEATURES_CSV  = "data/processed/features.csv"
MODEL_PATH    = "models/bayesian_model.pkl"
NETWORK_IMG   = "outputs/network_graph.png"


#  SETUP — Run once to train model
def setup():
    """Preprocess data and train model if not already done."""

    if not os.path.exists(FEATURES_CSV):
        print("First time setup: Processing dataset\n")
        load_and_process(FAKE_CSV, REAL_CSV, FEATURES_CSV)
    else:
        print("Processed features found. Skipping preprocessing.")

    if not os.path.exists(MODEL_PATH):
        print("Training Bayesian Network\n")
        build_and_train(FEATURES_CSV, MODEL_PATH)
    else:
        print("Trained model found. Skipping training.")

    print()


#  MENU

def print_menu():
    print("=" * 45)
    print("   FAKE NEWS DETECTOR — Bayesian Network")
    print("=" * 45)
    print()
    print("  Choose mode:")
    print("    1. Analyze a news article")
    print("    2. Show Bayesian Network diagram")
    print("    3. Exit")
    print()


def option_analyze(model):
    """Option 1 — Let user type a news article and get prediction."""
    while True:
        print("─" * 45)
        print("Enter news short article:")
        print("(type 'back' to return to menu)")
        print()
        text = input("> ").strip()
        print()

        if text.lower() == 'back':
            break

        if len(text) < 5:
            print("Please enter a longer text.\n")
            continue

        display_result(text, model)

        again = input("Want to analyze another article? (y/n): ").strip().lower()
        print()
        if again != 'y':
            break


def option_network():
    """Option 2 — Show the Bayesian Network diagram."""
    print()
    print("Opening Bayesian Network diagram")
    show_network(MODEL_PATH, NETWORK_IMG)


#  MAIN LOOP

def main():

    setup()

    # Load model into memory
    print("Loading model")
    model = load_model(MODEL_PATH)
    print("Model ready\n")

    while True:
        print_menu()
        choice = input("Enter choice: ").strip()
        print()

        if choice == '1':
            option_analyze(model)

        elif choice == '2':
            option_network()

        elif choice == '3':
            print("Exit")
            break

        else:
            print("Invalid choice. Please enter 1, 2, or 3.\n")


if __name__ == "__main__":
    main()