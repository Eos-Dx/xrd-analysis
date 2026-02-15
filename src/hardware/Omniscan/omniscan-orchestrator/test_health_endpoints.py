"""
Test script for /api/health and /api/connection/status endpoints

Prerequisites:
1. Hardware server running on localhost:50051
2. Orchestrator server running on localhost:8080
3. Valid session (login first)
"""

import requests
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8080"

def print_json(data: Dict[Any, Any], title: str = ""):
    """Pretty print JSON response"""
    if title:
        print(f"\n{'='*60}")
        print(f"{title}")
        print('='*60)
    print(json.dumps(data, indent=2))

def test_connection_status():
    """Test /api/connection/status endpoint (no auth required)"""
    print("\n🔍 Testing /api/connection/status...")
    
    response = requests.get(f"{BASE_URL}/api/connection/status")
    print(f"Status Code: {response.status_code}")
    
    data = response.json()
    print_json(data, "Connection Status")
    
    if data.get("connected"):
        print("✅ Hardware server is connected")
    else:
        print(f"❌ Hardware server not connected: {data.get('error')}")
    
    return data.get("connected", False)

def test_login() -> str:
    """Login and get session ID"""
    print("\n🔐 Logging in...")
    
    response = requests.post(
        f"{BASE_URL}/api/auth/login",
        json={"username": "operator1", "password": "test123"}
    )
    
    data = response.json()
    if data.get("success"):
        session_id = data.get("session_id")
        print(f"✅ Login successful, session_id: {session_id[:16]}...")
        return session_id
    else:
        print(f"❌ Login failed: {data.get('error')}")
        return None

def test_health_endpoint(session_id: str):
    """Test /api/health endpoint (requires auth)"""
    print("\n🏥 Testing /api/health...")
    
    headers = {"x-session-id": session_id}
    response = requests.get(f"{BASE_URL}/api/health", headers=headers)
    
    print(f"Status Code: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print_json(data, "System Health")
        
        # Validate structure
        print("\n📊 Validation:")
        
        # Check system state
        print(f"  System State: {data.get('state')}")
        
        # Check devices
        devices = ["pdu", "gpio", "detector", "motion"]
        for device in devices:
            if device in data:
                device_data = data[device]
                powered = device_data.get("powered", False)
                status = device_data.get("status", "UNKNOWN")
                
                symbol = "✅" if powered else "⚪"
                print(f"  {symbol} {device.upper()}: {status} (powered={powered})")
                
                # Check PDU outputs
                if device == "pdu" and "outputs" in device_data:
                    outputs = device_data["outputs"]
                    print(f"      Outputs: {outputs}")
        
        # Check interlocks
        interlocks = data.get("interlocks", {})
        overall_safe = interlocks.get("overall_safe", False)
        print(f"  {'✅' if overall_safe else '❌'} Interlocks Overall Safe: {overall_safe}")
        print(f"      Key Switch: {interlocks.get('key_switch')}")
        print(f"      Enable Button: {interlocks.get('enable_button')}")
        
        return True
    else:
        print(f"❌ Failed to get health: {response.text}")
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("Testing Omniscan Orchestrator Health Endpoints")
    print("="*60)
    
    # Test 1: Connection status (no auth)
    connected = test_connection_status()
    if not connected:
        print("\n⚠️  Hardware server not connected. Some tests may fail.")
    
    # Test 2: Login
    session_id = test_login()
    if not session_id:
        print("\n❌ Cannot proceed without valid session")
        return
    
    # Test 3: Health endpoint
    success = test_health_endpoint(session_id)
    
    # Summary
    print("\n" + "="*60)
    if success:
        print("✅ All tests passed!")
    else:
        print("❌ Some tests failed")
    print("="*60)

if __name__ == "__main__":
    main()
