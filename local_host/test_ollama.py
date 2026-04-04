import sys
import io

# Force UTF-8 encoding for stdout to handle emojis in LLM response
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from medical import ask

def test_ollama():
    print("Testing Medical RAG with Ollama...")
    query = "I have a severe headache and blurred vision."
    print(f"Query: {query}")
    try:
        response = ask(query)
        print("\n--- AI Response ---")
        # Ensure we can print the response even if there are weird characters
        print(response)
        print("--- End of Response ---\n")
        print("Verification Successful!")
    except Exception as e:
        # Avoid printing the emoji in the exception message itself if possible
        print(f"Verification Failed: {str(e).encode('ascii', 'ignore').decode()}")

if __name__ == "__main__":
    test_ollama()
