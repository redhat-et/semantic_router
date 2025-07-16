#!/usr/bin/env python3
"""
Temperature Testing Script for Semantic Router
Tests different temperature values to show the effect on LLM responses.
"""

import requests
import json
import time

def test_temperature(message, temperature, endpoint="http://localhost:8801/v1/chat/completions"):
    """Send a request with specific temperature setting"""
    
    request_body = {
        "model": "mistral-small3.1",
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": False,
        "temperature": temperature,
        "max_tokens": 150
    }
    
    try:
        response = requests.post(
            endpoint, 
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            return result.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
        else:
            return f"Error: {response.status_code} - {response.text}"
            
    except Exception as e:
        return f"Request failed: {str(e)}"

def main():
    print("🌡️  Temperature Testing for Semantic Router")
    print("=" * 50)
    
    # Test message that should show creativity differences
    test_message = "Write a creative opening line for a story about a robot learning to paint."
    
    temperatures = [0.0, 0.3, 0.7, 1.0, 1.5]
    
    print(f"Test Question: {test_message}")
    print("\n" + "=" * 50)
    
    for temp in temperatures:
        print(f"\n🌡️  Temperature: {temp}")
        print("-" * 30)
        
        response = test_temperature(test_message, temp)
        print(f"Response: {response}")
        
        # Add delay between requests
        time.sleep(1)
    
    print("\n" + "=" * 50)
    print("Temperature Test Complete!")
    print("\nObservations:")
    print("• Lower temperatures (0.0-0.3): More consistent, focused responses")
    print("• Medium temperatures (0.7): Balanced creativity and coherence") 
    print("• Higher temperatures (1.0+): More creative but potentially less coherent")

if __name__ == "__main__":
    main() 