import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Supabase PostgreSQL connection string
SUPABASE_CONNECTION_STRING = "postgresql://postgres:obstraatherealagent2027@db.ckjpcnkbroakxjjnyinp.supabase.co:5432/postgres"

# Use Supabase PostgreSQL!
SQLALCHEMY_DATABASE_URL = SUPABASE_CONNECTION_STRING
engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
