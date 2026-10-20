"""Upload ESLint analysis from the trusted master build using GitHub's REST API."""
import base64
import gzip
import json
import os
from pathlib import Path
import time
import urllib.request

if os.environ.get("CHANGE_ID") or os.environ.get("BRANCH_NAME") != "master":
    raise ValueError("Code scanning publication requires a trusted master build")


def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request("https://api.github.com/repos/Frotty/war3-model/" + path, data=data,
                                 headers={"Authorization": "Bearer " + os.environ["GITHUB_TOKEN"],
                                          "Accept": "application/vnd.github+json",
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as response:
        return json.load(response)


sarif = Path("eslint-results.sarif").read_bytes()
json.loads(sarif)  # Reject incomplete output before upload.
result = request("code-scanning/sarifs", {"commit_sha": os.environ["GIT_COMMIT"],
                 "ref": "refs/heads/master", "sarif": base64.b64encode(gzip.compress(sarif)).decode("ascii")})
for attempt in range(60):
    status = request("code-scanning/sarifs/" + result["id"])
    state = status["processing_status"]
    if state == "complete":
        print("ESLint SARIF processed by GitHub")
        break
    if state == "failed":
        raise RuntimeError("GitHub rejected SARIF: " + json.dumps(status.get("errors", [])))
    time.sleep(5)
else:
    raise TimeoutError("GitHub did not finish processing the ESLint report within five minutes")
