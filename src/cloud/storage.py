"""A Cloud Storage bucket over its JSON API, with the few calls the reviews need.

Preconditions are what make the review job safe to deliver twice: ``put(..., only_if_absent=True)`` succeeds for
exactly one writer, and ``put(..., if_generation=g)`` only replaces the version that was read. Uses ``google-auth``
and ``requests`` (both already dependencies); no Cloud client library.
"""

from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import quote

API = "https://storage.googleapis.com/storage/v1/b"
UPLOAD = "https://storage.googleapis.com/upload/storage/v1/b"
SCOPE = "https://www.googleapis.com/auth/cloud-platform"
TIMEOUT_S = 120


def session(credentials_file: str | Path | None = None):
    """An authorised HTTP session: from a service-account key file, or the environment's default credentials
    (on Cloud Run and Compute Engine that is the attached service account)."""
    import google.auth
    from google.auth.transport.requests import AuthorizedSession
    from google.oauth2 import service_account

    if credentials_file:
        creds = service_account.Credentials.from_service_account_file(str(credentials_file), scopes=[SCOPE])
    else:
        creds, _ = google.auth.default(scopes=[SCOPE])
    return AuthorizedSession(creds)


class StorageError(RuntimeError):
    """Cloud Storage answered with something other than success, not-found or a failed precondition."""


class Bucket:
    def __init__(self, name: str, http):
        self.name, self.http = name, http

    def _object(self, key: str) -> str:
        return f"{API}/{self.name}/o/{quote(key, safe='')}"

    def _check(self, r, ok=(200,), soft=()) -> bool:
        if r.status_code in ok:
            return True
        if r.status_code in soft:
            return False
        raise StorageError(f"storage answered {r.status_code}")

    def stat(self, key: str) -> dict | None:
        """``{"generation", "size", "updated"}`` of the object, or None if there is none."""
        r = self.http.get(self._object(key), params={"fields": "generation,size,updated"}, timeout=TIMEOUT_S)
        return r.json() if self._check(r, soft=(404,)) else None

    def get(self, key: str) -> bytes | None:
        r = self.http.get(self._object(key), params={"alt": "media"}, timeout=TIMEOUT_S)
        return r.content if self._check(r, soft=(404,)) else None

    def get_json(self, key: str):
        data = self.get(key)
        return json.loads(data) if data is not None else None

    def download(self, key: str, dest: Path) -> bool:
        with self.http.get(self._object(key), params={"alt": "media"}, timeout=TIMEOUT_S, stream=True) as r:
            if not self._check(r, soft=(404,)):
                return False
            with open(dest, "wb") as fh:
                for chunk in r.iter_content(1 << 20):
                    fh.write(chunk)
        return True

    def put(self, key: str, data, only_if_absent: bool = False, if_generation: str | int | None = None,
            content_type: str = "application/octet-stream") -> bool:
        """Write the object from bytes or an open binary file. False when the precondition did not hold: the
        object already exists (``only_if_absent``) or is not the version ``if_generation`` any more."""
        params = {"uploadType": "media", "name": key}
        if only_if_absent:
            params["ifGenerationMatch"] = "0"
        elif if_generation is not None:
            params["ifGenerationMatch"] = str(if_generation)
        r = self.http.post(f"{UPLOAD}/{self.name}/o", params=params, data=data, headers={"Content-Type": content_type},
                           timeout=TIMEOUT_S)
        return self._check(r, soft=(412,))

    def put_json(self, key: str, value, **kw) -> bool:
        return self.put(key, json.dumps(value, allow_nan=False).encode(), content_type="application/json", **kw)

    def put_file(self, key: str, path: Path, only_if_absent: bool = False) -> bool:
        with open(path, "rb") as fh:
            return self.put(key, fh, only_if_absent=only_if_absent)

    def delete(self, key: str) -> bool:
        return self._check(self.http.delete(self._object(key), timeout=TIMEOUT_S), ok=(200, 204), soft=(404,))

    def list(self, prefix: str) -> list[str]:
        keys, token = [], None
        while True:
            params = {"prefix": prefix, "fields": "items(name),nextPageToken", **({"pageToken": token} if token else {})}
            r = self.http.get(f"{API}/{self.name}/o", params=params, timeout=TIMEOUT_S)
            self._check(r)
            body = r.json()
            keys += [item["name"] for item in body.get("items", [])]
            token = body.get("nextPageToken")
            if not token:
                return keys
