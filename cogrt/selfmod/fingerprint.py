from __future__ import annotations

import hashlib
import json
from typing import Any


def intervention_fingerprint(parent_sha256: str, method: str, spec: dict[str, Any]) -> str:
    payload = {"parent": parent_sha256, "method": method, "spec": spec}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
