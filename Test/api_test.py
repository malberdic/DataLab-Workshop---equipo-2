from fastapi.testclient import TestClient
from ..src.API.review_api  import app

client = TestClient(app)

def test_pipeline():
    r = client.post("/pipeline/com.spotify.music", params={"limit": 50})
    print(r.status_code)
    print(r.json())

test_pipeline()