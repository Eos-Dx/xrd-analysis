"""
Test script to verify state change notifications from hardware server.
Run this while toggling the key switch in the GPIO GUI to see if notifications arrive.
"""

from src.omniscan_orchestrator.grpc_client import OmniscanGrpcClient
import time

def test_notifications():
    print("🔌 Connecting to hardware server...")
    client = OmniscanGrpcClient(server_address="localhost:50051")
    
    print("📡 Subscribing to state updates...")
    stream = client.subscribe_to_state_updates()
    
    if stream is None:
        print("❌ Failed to subscribe!")
        return
    
    print("✅ Subscribed! Waiting for notifications...")
    print("💡 Toggle the key switch in the GPIO GUI to test...")
    print()
    
    try:
        for notification in stream:
            print(f"📢 Received: {notification.component} - {notification.change_type}")
            print(f"   Timestamp: {notification.timestamp.seconds if notification.timestamp else 'N/A'}")
            print()
    except KeyboardInterrupt:
        print("\n👋 Stopping...")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    test_notifications()
