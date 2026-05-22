import time
import datetime
import requests
import jwt
from config import settings

#objecct have funct mainly to extract installation id for calling endpoint in other services
class GithubAuth:
    def __init__(self,app_id:int, private_key_pem:str,installation_id:int):
        self.app_id = app_id
        self.private_key_pem = private_key_pem
        self.installation_id = installation_id
        self._cached_token: dict | None = None

    def mint_jwt(self)->str:
        now=int(time.time())
        payload={
            "iat":now-60,
            "exp":now+540,
            "iss":str(self.app_id)
        }
        return jwt.encode(
            payload,
            self.private_key_pem,
            algorithm="RS256"
        )

    def _fetch_installation_token(self)->dict:
        token=self.mint_jwt()
        resp=requests.post(
            f"https://api.github.com/app/installations/{self.installation_id}/access_tokens",
            headers={
                "Authorization": f"Bearer {token}",
                "Accept": "application/vnd.github+json",
            },
        )
        resp.raise_for_status()
        return resp.json()

    def _token_expires_soon(self) -> bool:
        if not self._cached_token:
            return True
        expires_at = datetime.datetime.strptime(
            self._cached_token["expires_at"], "%Y-%m-%dT%H:%M:%SZ"
        )
        return (expires_at - datetime.datetime.utcnow()).total_seconds() < 300

    def get_installation_token(self)->str:
        if self._token_expires_soon():
            self._cached_token=self._fetch_installation_token()
        return self._cached_token["token"]

def make_auth(installation_id:int)->GithubAuth:
    return GithubAuth(
        app_id=settings.github_app_id,
        private_key_pem=settings.github_private_key_pem,
        installation_id=installation_id
    )

