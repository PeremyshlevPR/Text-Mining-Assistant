from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from conf import settings

engine = create_engine(settings.DB_URL)
Session = sessionmaker(engine, expire_on_commit=False)
