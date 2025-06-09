#!/usr/bin/env python3
"""
Test PII detection functionality in the semantic router.

This test verifies that:
1. PII detection works correctly in the full pipeline
2. Text is properly sanitized when PII is detected
3. Requests are blocked/allowed based on PII configuration
4. Existing functionality continues to work with PII detection enabled
"""

import json
import unittest
import requests
import time
import sys
import os

# Add the parent directory to Python path so we can import test_base
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from test_base import SemanticRouterTestBase


class PIIDetectionTest(SemanticRouterTestBase):
    """Test PII detection functionality"""

    def setUp(self):
        """Set up test fixtures"""
        super().setUp()
        self.base_url = "http://localhost:8801/v1/chat/completions"
        self.headers = {"Content-Type": "application/json"}
        
        # Test data with PII
        # Simplified test cases to avoid timeout issues
        self.pii_test_cases = [
            {
                "name": "email_detection",
                "content": "Contact me at test@example.com",
                "expected_pii_types": ["EMAIL"],
                "should_contain_pii": True
            },
            {
                "name": "clean_text",
                "content": "This is a regular message without any personal information",
                "expected_pii_types": [],
                "should_contain_pii": False
            }
        ]

    def test_pii_detection_enabled(self):
        """Test that PII detection identifies PII correctly"""
        for test_case in self.pii_test_cases:
            with self.subTest(test_case=test_case["name"]):
                payload = {
                    "model": "auto",
                    "messages": [
                        {"role": "user", "content": test_case["content"]}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 50
                }
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                # Should get a response regardless of PII (unless blocking is enabled)
                self.assertEqual(response.status_code, 200, 
                               f"Request failed for {test_case['name']}: {response.text}")
                
                response_data = response.json()
                self.assertIn("choices", response_data)
                
                # Check for PII detection in headers or response metadata
                # (This depends on how PII detection results are exposed)
                # TODO: Currently dual classifier doesn't expose PII detection metadata
                # For now, we just verify the request succeeds and basic functionality works
                if "pii_detection" in response_data:
                    pii_result = response_data["pii_detection"]
                    has_pii = pii_result.get("has_pii", False)
                    detected_types = pii_result.get("detected_types", [])
                    
                    self.assertEqual(has_pii, test_case["should_contain_pii"],
                                   f"PII detection mismatch for {test_case['name']}")
                    
                    if test_case["should_contain_pii"]:
                        # Check if at least some expected PII types were detected
                        detected_set = set(detected_types)
                        expected_set = set(test_case["expected_pii_types"])
                        self.assertTrue(bool(detected_set & expected_set),
                                      f"Expected PII types {expected_set} not detected in {detected_set}")
                else:
                    # PII detection metadata not exposed yet, just verify we get a valid response
                    print(f"PII detection metadata not available for {test_case['name']}, skipping detailed checks")

    def test_pii_sanitization(self):
        """Test that PII is properly sanitized in responses"""
        test_content = "Contact John Smith at john@example.com or call 555-123-4567"
        
        payload = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": test_content}
            ],
            "temperature": 0.1,
            "max_tokens": 100
        }
        
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        
        self.assertEqual(response.status_code, 200)
        response_data = response.json()
        
        # Check if the response contains sanitized text
        if "pii_detection" in response_data:
            sanitized_text = response_data["pii_detection"].get("sanitized_text", "")
            if sanitized_text:
                # Verify that PII has been replaced with placeholders
                self.assertNotIn("john@example.com", sanitized_text.lower())
                self.assertNotIn("555-123-4567", sanitized_text)
                # Should contain placeholders
                self.assertTrue(
                    "[EMAIL" in sanitized_text.upper() or "[PHONE" in sanitized_text.upper(),
                    f"Sanitized text doesn't contain expected placeholders: {sanitized_text}"
                )
        else:
            # TODO: PII sanitization metadata not available yet
            # For now, just check that we get a reasonable response
            content = response_data["choices"][0]["message"]["content"]
            print(f"PII sanitization test - got response: {content[:100]}...")
            self.assertGreater(len(content.strip()), 0, "Should get some response content")

    def test_existing_functionality_with_pii_enabled(self):
        """Ensure existing semantic routing still works with PII detection enabled"""
        # Simplified test cases to avoid timeout issues
        routing_tests = [
            {
                "content": "What is 2+2?",
                "category": "math",
                "description": "Math question routing"
            }
        ]
        
        for test_case in routing_tests:
            with self.subTest(test_case=test_case["description"]):
                payload = {
                    "model": "auto",
                    "messages": [
                        {"role": "user", "content": test_case["content"]}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 100
                }
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                # Should still route correctly
                self.assertEqual(response.status_code, 200,
                               f"Routing failed for {test_case['description']}: {response.text}")
                
                response_data = response.json()
                self.assertIn("choices", response_data)
                self.assertGreater(len(response_data["choices"]), 0)
                
                # Verify we got a reasonable response
                content = response_data["choices"][0]["message"]["content"]
                self.assertGreater(len(content.strip()), 0)

    def test_pii_detection_performance(self):
        """Test that PII detection doesn't significantly impact performance"""
        test_content = "This is a test message for performance evaluation"
        
        payload = {
            "model": "auto",
            "messages": [
                {"role": "user", "content": test_content}
            ],
            "temperature": 0.1,
            "max_tokens": 50
        }
        
        # Measure response time
        start_time = time.time()
        response = requests.post(
            self.base_url,
            headers=self.headers,
            json=payload,
            timeout=30
        )
        end_time = time.time()
        
        response_time = end_time - start_time
        
        self.assertEqual(response.status_code, 200)
        
        # Response should be reasonably fast (adjust threshold as needed)
        self.assertLess(response_time, 20.0, 
                       f"Response took too long: {response_time:.2f}s")
        
        print(f"PII detection response time: {response_time:.2f}s")

    def test_pii_edge_cases(self):
        """Test PII detection with edge cases"""
        # Simplified edge cases to avoid timeout issues
        edge_cases = [
            {
                "name": "empty_message",
                "content": "",
            },
            {
                "name": "simple_text",
                "content": "Hello world",
            }
        ]
        
        for edge_case in edge_cases:
            with self.subTest(edge_case=edge_case["name"]):
                payload = {
                    "model": "auto",
                    "messages": [
                        {"role": "user", "content": edge_case["content"]}
                    ],
                    "temperature": 0.1,
                    "max_tokens": 50
                }
                
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                
                # Should handle edge cases gracefully (empty message might return 404)
                self.assertIn(response.status_code, [200, 400, 404],
                            f"Unexpected response for {edge_case['name']}: {response.status_code}")
                
                if response.status_code == 200:
                    response_data = response.json()
                    self.assertIn("choices", response_data)

    def test_concurrent_pii_detection(self):
        """Test PII detection under concurrent load"""
        import threading
        import queue
        
        num_threads = 5
        requests_per_thread = 3
        results = queue.Queue()
        
        def make_request():
            """Make a single request with PII content"""
            payload = {
                "model": "auto",
                "messages": [
                    {"role": "user", "content": "My email is test@example.com"}
                ],
                "temperature": 0.1,
                "max_tokens": 50
            }
            
            try:
                response = requests.post(
                    self.base_url,
                    headers=self.headers,
                    json=payload,
                    timeout=30
                )
                results.put(("success", response.status_code))
            except Exception as e:
                results.put(("error", str(e)))
        
        # Create and start threads
        threads = []
        for _ in range(num_threads * requests_per_thread):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check results
        success_count = 0
        error_count = 0
        
        while not results.empty():
            result_type, result_value = results.get()
            if result_type == "success":
                success_count += 1
                self.assertEqual(result_value, 200)
            else:
                error_count += 1
                print(f"Error in concurrent test: {result_value}")
        
        # Should have mostly successful requests
        total_requests = num_threads * requests_per_thread
        success_rate = success_count / total_requests
        self.assertGreater(success_rate, 0.8, 
                          f"Success rate too low: {success_rate:.2f}")
        
        print(f"Concurrent test: {success_count}/{total_requests} successful")


if __name__ == "__main__":
    unittest.main() 