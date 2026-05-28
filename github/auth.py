import time
import datetime
import requests
import jwt
from config import settings
import redis

#objecct have funct mainly to extract installation id for calling endpoint in other services
class GithubAuth:
    def __init__(self, app_id: int, private_key_pem: str, installation_id: int, redis_client: redis.Redis):
        self.app_id = app_id
        self.private_key_pem = private_key_pem
        self.installation_id = installation_id
        self.redis_client = redis_client

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

    def get_installation_token(self)->str:
        key = f"github:token:{self.installation_id}"
        cached=self.redis_client.get(key)
        if cached:
            return cached.decode() if isinstance(cached, bytes) else cached
        token_data=self._fetch_installation_token()
        self.redis_client.setex(key,55*60,token_data["token"])
        return token_data["token"]

def make_auth(installation_id:int)->GithubAuth:
    r=redis.from_url(settings.redis_url)
    return GithubAuth(
        app_id=settings.github_app_id,
        private_key_pem=settings.github_private_key_pem,
        installation_id=installation_id,
        redis_client=r
    )

