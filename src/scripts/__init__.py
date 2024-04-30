from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

test_engine = create_engine(
    "sqlite:///../../data/database/text_mining.db"
)
Session = sessionmaker(test_engine)
