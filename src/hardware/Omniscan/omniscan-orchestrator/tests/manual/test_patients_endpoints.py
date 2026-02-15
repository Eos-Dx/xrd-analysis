import os
import uuid
from pathlib import Path

from fastapi.testclient import TestClient

from omniscan_orchestrator import rest_server
from omniscan_orchestrator.database import OrchestratorDatabase, UserSession
from omniscan_orchestrator.auth_manager import AuthenticationManager


class DummyGrpc:
    def get_gpio_state(self):
        # Pretend key switch is ON
        return {"key_switch_on": True, "interlocks": {"overall_safe": True}}


def run():
    # Use a temp DB under project data/test
    project_root = Path(__file__).resolve().parents[2]
    db_path = project_root / "data" / "test_orchestrator.db"
    if db_path.exists():
        db_path.unlink()

    db = OrchestratorDatabase(str(db_path))
    grpc = DummyGrpc()
    auth = AuthenticationManager(grpc, db)

    # Seed a session
    session_id = str(uuid.uuid4())
    sess = UserSession(
        session_id=session_id,
        user_id="OP001",
        role="operator",
        login_time=__import__("datetime").datetime.utcnow(),
        logout_time=None,
        last_activity=__import__("datetime").datetime.utcnow(),
    )
    auth.active_sessions[session_id] = sess
    db.record_login(sess.user_id, session_id, sess.role)

    # Inject globals
    rest_server.db = db
    rest_server.grpc_client = grpc
    rest_server.auth_manager = auth

    client = TestClient(rest_server.app)
    headers = {"x-session-id": session_id}

    # 1) Create patient
    payload = {
        "first_name": "Alice",
        "last_name": "Smith",
        "date_of_birth": "1980-01-02T00:00:00",
        "medical_record_number": "MRN-TEST-1234",
    }
    r = client.post("/api/patients", headers=headers, json=payload)
    assert r.status_code == 200, r.text
    created = r.json()

    # 2) Search by MRN
    r2 = client.get("/api/patients/search", headers=headers, params={"mrn": payload["medical_record_number"]})
    assert r2.status_code == 200, r2.text
    searched = r2.json()
    assert searched["patient_id"] == created["patient_id"]

    # 3) Load by ID
    r3 = client.get(f"/api/patients/{created['patient_id']}", headers=headers)
    assert r3.status_code == 200, r3.text
    loaded = r3.json()

    return {
        "created": created,
        "searched": searched,
        "loaded": loaded,
    }


if __name__ == "__main__":
    import json
    res = run()
    print(json.dumps(res, indent=2))
