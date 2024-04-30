from typing import Literal

import logging
logger = logging.getLogger(__name__)

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from db import models
from .database import Session
import utils

def create_user(username: str, password: str, role: Literal["user", "admin"] = "user") -> models.User:
    hashed_password, salt = utils.hash_password(password)

    with Session() as session, session.begin():
        user = models.User(
            username = username,
            password_hash=hashed_password,
            password_salt=salt,
            role=role
        )
        session.add(user)
        logger.info(f'User {username} successfully inserted to database. id = {user.id}')
    return user

def get_users(return_one=True, **kwargs):
    with Session() as session:
        result = session.scalars(
            select(models.User).filter_by(**kwargs)
        )
        
    if return_one:
        return result.first()
    return result.all()
    