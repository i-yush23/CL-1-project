
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hinglish.pipeline_runner import run_pipeline

if __name__ == "__main__":
    demo_text = "Mujhe lagta hai ki tomorrow's meeting cancel ho jayegi, but let's see kya hota hai. I'll call you later, bye!"
    print("\n" + "="*50)
    print(" HINGLISH NLP DEMONSTRATION ".center(50, "="))
    print("="*50)
    print(f"INPUT: {demo_text}")
    
    run_pipeline(demo_text)
    
    print("="*50)
