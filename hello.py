from langchain_google_genai import GoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

# Initialize the LLM
llm = GoogleGenerativeAI(model="gemini-2.5-flash")

# Stream and collect
full_response = ""
for chunk in llm.stream("Once upon a time, a library called LangChain"):
    print(chunk, end="", flush=True)   # show in terminal
    full_response += chunk             # store in string

print("\n---\nDone!")
print("\nFinal collected response:\n", full_response)

# Save to file
with open("llm_output.txt", "w", encoding="utf-8") as f:
    f.write(full_response)

print("\n Response saved to llm_output.txt")
