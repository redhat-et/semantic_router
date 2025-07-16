#!/usr/bin/env python3
"""
Category-Specific Temperature Testing Script for Semantic Router
Tests that different categories use their configured temperature values.
"""

import requests
import json
import time

def test_category_temperature(message, expected_category, expected_temperature, endpoint="http://localhost:8801/v1/chat/completions"):
    """Send a request and check if the correct temperature is used for the category"""
    
    request_body = {
        "model": "mistral-small3.1",
        "messages": [
            {"role": "user", "content": message}
        ],
        "stream": False,
        "temperature": 0.7  # This should be overridden by category-specific temperature
    }
    
    try:
        response = requests.post(
            endpoint, 
            json=request_body,
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            # Check headers for classification info
            headers = response.headers
            category_header = headers.get('x-category', 'Unknown')
            temp_header = headers.get('x-temperature', 'Unknown')
            
            result = response.json()
            response_content = result.get('choices', [{}])[0].get('message', {}).get('content', 'No response')
            
            return {
                'success': True,
                'category': category_header,
                'temperature': temp_header,
                'response': response_content[:100] + "..." if len(response_content) > 100 else response_content,
                'expected_category': expected_category,
                'expected_temperature': expected_temperature
            }
        else:
            return {
                'success': False,
                'error': f"HTTP {response.status_code}: {response.text}",
                'expected_category': expected_category,
                'expected_temperature': expected_temperature
            }
            
    except Exception as e:
        return {
            'success': False,
            'error': f"Request failed: {str(e)}",
            'expected_category': expected_category,
            'expected_temperature': expected_temperature
        }

def main():
    print("🌡️  Category-Specific Temperature Testing for Semantic Router")
    print("=" * 60)
    
    # Test cases with expected categories and temperatures
    test_cases = [
        {
            'message': 'Write a Python function to sort a list',
            'expected_category': 'Programming',
            'expected_temperature': 0.3,
            'description': 'Programming Question'
        },
        {
            'message': 'What happened during World War 2?',
            'expected_category': 'History',
            'expected_temperature': 0.5,
            'description': 'History Question'
        },
        {
            'message': 'What are the symptoms of diabetes?',
            'expected_category': 'Health',
            'expected_temperature': 0.5,
            'description': 'Health Question'
        },
        {
            'message': 'What is the derivative of x^2?',
            'expected_category': 'Math',
            'expected_temperature': 0.3,
            'description': 'Mathematics Question'
        },
        {
            'message': 'Tell me an interesting story about a robot',
            'expected_category': 'General',
            'expected_temperature': 0.9,
            'description': 'General Creative Question'
        }
    ]
    
    print(f"Testing {len(test_cases)} category-temperature combinations...")
    print("\n" + "=" * 60)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n🧪 Test {i}: {test_case['description']}")
        print(f"Question: {test_case['message']}")
        print(f"Expected: {test_case['expected_category']} category, temp {test_case['expected_temperature']}")
        print("-" * 40)
        
        result = test_category_temperature(
            test_case['message'],
            test_case['expected_category'],
            test_case['expected_temperature']
        )
        
        if result['success']:
            category_match = result['category'].lower() == test_case['expected_category'].lower()
            temp_match = abs(float(result.get('temperature', 0)) - test_case['expected_temperature']) < 0.1
            
            print(f"✅ Request successful")
            print(f"📂 Category: {result['category']} {'✅' if category_match else '❌'}")
            print(f"🌡️  Temperature: {result.get('temperature', 'Not reported')} {'✅' if temp_match else '❌'}")
            print(f"📝 Response: {result['response']}")
            
            if category_match and temp_match:
                print("🎯 PERFECT MATCH!")
            elif category_match:
                print("⚠️  Category correct, but temperature info missing/incorrect")
            else:
                print("❌ Category mismatch")
                
        else:
            print(f"❌ Request failed: {result['error']}")
        
        print()
        # Add delay between requests
        time.sleep(1)
    
    print("=" * 60)
    print("Category-Temperature Test Complete!")
    print("\n📊 Expected Temperature Configuration:")
    print("• Programming: 0.3 (Low - for precise code generation)")
    print("• Mathematics: 0.3 (Low - for precise calculations)")
    print("• History: 0.5 (Medium-low - for factual information)")
    print("• Health: 0.5 (Medium-low - for medical accuracy)")
    print("• General: 0.9 (High - for creative responses)")
    print("\n💡 Note: Backend logs will show temperature assignment")

if __name__ == "__main__":
    main() 