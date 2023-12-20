# Test the API which may be run with pytest

from fastapi.testclient import TestClient
from .main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/hello_world")
    assert response.status_code == 200
    assert response.json() == "Hello world!"
