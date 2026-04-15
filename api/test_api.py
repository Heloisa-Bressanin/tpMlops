"""
Test script for the Taxi Duration Prediction API
"""

import requests
import json
import time

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Check")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/")
        print(f"Status: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_single_prediction():
    """Test single prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Single Prediction")
    print("="*60)
    
    payload = {
        "pickup_datetime": "2016-03-14 17:24:55",
        "pickup_latitude": 40.7614,
        "pickup_longitude": -73.9776,
        "dropoff_latitude": 40.7489,
        "dropoff_longitude": -73.9680,
        "passenger_count": 1,
        "vendor_id": 1
    }
    
    try:
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        print(f"Status: {response.status_code}")
        result = response.json()
        print(f"\nPrediction Result:")
        print(f"  Duration: {result['predicted_duration_minutes']:.1f} minutes")
        print(f"  Duration: {result['predicted_duration_seconds']:.1f} seconds")
        print(f"  ID: {result['prediction_id']}")
        print(f"  Timestamp: {result['timestamp']}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_batch_prediction():
    """Test batch prediction endpoint"""
    print("\n" + "="*60)
    print("Testing Batch Prediction")
    print("="*60)
    
    payloads = [
        {
            "pickup_datetime": "2016-03-14 17:24:55",
            "pickup_latitude": 40.7614,
            "pickup_longitude": -73.9776,
            "dropoff_latitude": 40.7489,
            "dropoff_longitude": -73.9680,
            "passenger_count": 1,
            "vendor_id": 1
        },
        {
            "pickup_datetime": "2016-03-14 10:15:30",
            "pickup_latitude": 40.7505,
            "pickup_longitude": -73.9972,
            "dropoff_latitude": 40.7614,
            "dropoff_longitude": -73.9776,
            "passenger_count": 2,
            "vendor_id": 2
        },
        {
            "pickup_datetime": "2016-03-14 22:50:00",
            "pickup_latitude": 40.6501,
            "pickup_longitude": -73.9496,
            "dropoff_latitude": 40.7589,
            "dropoff_longitude": -73.9851,
            "passenger_count": 3,
            "vendor_id": 1
        }
    ]
    
    try:
        response = requests.post(f"{BASE_URL}/predict-batch", json=payloads)
        print(f"Status: {response.status_code}")
        results = response.json()
        print(f"\nPredictions made: {len(results)}")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['predicted_duration_minutes']:.1f} min ({result['predicted_duration_seconds']:.1f}s)")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_stats():
    """Test statistics endpoint"""
    print("\n" + "="*60)
    print("Testing Statistics")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/stats")
        print(f"Status: {response.status_code}")
        stats = response.json()
        print(f"\nPrediction Statistics:")
        print(f"  Total predictions: {stats['total_predictions']}")
        print(f"  Average duration: {stats['average_duration_seconds']/60:.1f} min")
        print(f"  Min duration: {stats['min_duration_seconds']/60:.1f} min")
        print(f"  Max duration: {stats['max_duration_seconds']/60:.1f} min")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_history():
    """Test history endpoint"""
    print("\n" + "="*60)
    print("Testing Prediction History")
    print("="*60)
    try:
        response = requests.get(f"{BASE_URL}/history?limit=5")
        print(f"Status: {response.status_code}")
        history = response.json()
        print(f"\nRecent predictions: {len(history)}")
        for record in history:
            print(f"  ID #{record['id']}: {record['predicted_duration_minutes']:.1f} min")
            print(f"    Pickup: {record['pickup_datetime']}")
        return True
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*60)
    print("NYC TAXI TRIP DURATION PREDICTION API - TEST SUITE")
    print("="*60)
    
    results = {
        "Health Check": test_health(),
        "Single Prediction": test_single_prediction(),
        "Batch Prediction": test_batch_prediction(),
        "Statistics": test_stats(),
        "History": test_history()
    }
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    for test_name, result in results.items():
        status = "✓" if result else "✗"
        print(f"  {status} {test_name}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    print("Waiting 2 seconds for API to be ready...")
    time.sleep(2)
    run_all_tests()
