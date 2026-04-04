import ollama
def pull_tiny():
    print("Pulling tinyllama for emergency speed...")
    try:
        ollama.pull('tinyllama')
        print("Success!")
    except Exception as e:
        print(f"Error: {e}")
if __name__ == "__main__":
    pull_tiny()
