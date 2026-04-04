import ollama
import time

def test_phi3():
    print("Testing Phi-3 connectivity...")
    prompt = "Tell me one symptom of the common cold. Return ONLY the symptom name."
    start_time = time.time()
    try:
        response = ollama.generate(model='phi3', prompt=prompt)
        end_time = time.time()
        print(f"Response: {response['response']}")
        print(f"Time taken: {end_time - start_time:.2f} seconds")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    test_phi3()
