from typing import Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sqlmodel import Field, SQLModel, Session, create_engine, select

from fibsem.db.v3.models import User
from fibsem import config as cfg

# Create the database engine
def create_connection(database_path: str = cfg.DATABASE_PATH, echo: bool = False):
    # create a connection to the database
    engine = create_engine(f'sqlite:///{database_path}', echo=echo)

    return engine

def create_database(engine):
    # create the database
    SQLModel.metadata.create_all(engine)

engine = create_connection()

create_database(engine)

app = FastAPI()

origins = [
    "http://localhost:8001",  # The origin your HTML page is served from
    "http://127.0.0.1:8001",  # Also include the localhost version if necessary
    "http://0.0.0.0:8001"
]

# Add CORS middleware to allow cross-origin requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


# Create a user
@app.post("/users/")
def create_user(user: User):
    with Session(engine) as session:
        session.add(user)
        session.commit()
        session.refresh(user)
        return user

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
