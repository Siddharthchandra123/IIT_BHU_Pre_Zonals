import ollama
import sys

def pull_model(model_name):
    print(f"Pulling {model_name}...")
    try:
        ollama.pull(model_name)
        print(f"Successfully pulled {model_name}!")
    except Exception as e:
        print(f"Error pulling model: {e}")

if __name__ == "__main__":
    model = sys.argv[1] if len(sys.argv) > 1 else 'phi3'
    pull_model(model)
