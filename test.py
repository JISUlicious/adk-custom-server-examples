import urllib.request
import json

data = {
    "app_name": "agent",
    "user_id": "test_user",
    "session_id": "test-session-2",
    "new_message": {"role": "user", "parts": [{"text": "get some users list"}]},
}

req = urllib.request.Request(
    "http://localhost:8000/run",
    data=json.dumps(data).encode(),
    headers={
        "Content-Type": "application/json",
        "X-User-Id": "default_user",
        "X-User-Roles": "admin",
    },
)

try:
    response = urllib.request.urlopen(req)
    print("Status:", response.status)
    print("Response:", response.read().decode()[:300])
except urllib.error.HTTPError as e:
    print("Error:", e.code, e.reason)
    body = e.read().decode()
    print("Body:", body[:300])
