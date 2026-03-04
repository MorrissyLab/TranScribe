"""Diagnose model instantiation error for gemma-3-27b-it."""
import traceback

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain_core.messages import HumanMessage
    
    print("Attempting to create model gemma-3-27b-it...")
    llm = ChatGoogleGenerativeAI(model='gemma-3-27b-it', temperature=0.1)
    print("Model created successfully. Sending test message...")
    
    r = llm.invoke([HumanMessage(content="Say hello")])
    print(f"RESPONSE: {r.content}")
    
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()

print("\n--- Trying gemini-2.0-flash for comparison ---")
try:
    llm2 = ChatGoogleGenerativeAI(model='gemini-2.0-flash', temperature=0.1)
    print("gemini-2.0-flash model created OK")
    r2 = llm2.invoke([HumanMessage(content="Say hello")])
    print(f"RESPONSE: {r2.content}")
except Exception as e:
    print(f"FAILED: {type(e).__name__}: {e}")
    traceback.print_exc()
