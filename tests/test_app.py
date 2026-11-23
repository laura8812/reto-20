from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_ask_with_no_information():
    response = client.post("/ask", json={"question": "¿Cuál es la capital de Marte?"})
    assert response.status_code == 200
    assert "answer" in response.json()


def test_upload_and_ask():
    response = client.post(
        "/documents",
        files={"file": ("prueba.txt", b"La ayuda comunitaria funciona los martes.", "text/plain")},
    )
    assert response.status_code == 200
    assert response.json()["chunks"] >= 1

    response = client.post("/ask", json={"question": "¿Cuando funciona la ayuda?"})
    assert response.status_code == 200
    assert response.json()["sources"]
