import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from db.database import engine, Base
from db import models

print("Dropping all tables...")
Base.metadata.drop_all(bind=engine)
print("Creating all tables...")
Base.metadata.create_all(bind=engine)
print("Database reset complete!")
