from sqlalchemy.orm import DeclarativeBase, mapped_column
from sqlalchemy import String, DateTime, func
import uuid
from datetime import datetime
from sqlalchemy.orm import Mapped


class Base(DeclarativeBase):
    id: Mapped[str] = mapped_column(String, primary_key=True, index=True, default=lambda: str(uuid.uuid4()))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=func.current_timestamp())
    updated_at: Mapped[datetime] = mapped_column(DateTime,
                    default=func.current_timestamp(),
                    onupdate=func.current_timestamp())

class User(Base):
    __tablename__ = "users"

    username: Mapped[str] = mapped_column(String, unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String)
    role: Mapped[str] = mapped_column(String(20))
    