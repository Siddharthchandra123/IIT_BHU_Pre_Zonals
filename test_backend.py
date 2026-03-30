import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from frontend.medical import ask

print("Testing backend logic...")
response = ask("I have chest pain and I am coughing a lot")
print("\n--- RESPONSE ---")
print(response)
print("--- END ---")
