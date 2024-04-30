import jwt
from datetime import datetime, timezone, timedelta
import secrets
import hashlib
from conf import settings


def issue_token(user_id: str, role: str = "user") -> str:
    payload = {
        "user_id": user_id,
        "role": role,
        "exp": datetime.now(tz=timezone.utc) + timedelta(days=settings.JWT_EXPIRATION_TIME_DAYS),
        "iss": settings.JWT_ISSUER,
        "iat": datetime.now(tz=timezone.utc)
    }
    token = jwt.encode(payload, key=settings.JWT_SECRET_KEY)
    return token

def hash_password(password: str, salt=None):
    if salt is None:
        salt = secrets.token_hex(16)

    salted_password = password + salt
    hashed_password = hashlib.sha256(salted_password.encode()).hexdigest()
    
    return hashed_password, salt
    