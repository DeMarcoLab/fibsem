from sqlmodel import Session, SQLModel, create_engine, select
from fibsem import config as cfg

def create_connection(database_path: str = cfg.DATABASE_PATH, echo: bool = False):
    # create a connection to the database
    engine = create_engine(f'sqlite:///{database_path}', echo=echo)

    return engine

def create_database(engine):
    # create the database
    SQLModel.metadata.create_all(engine)


def get_session(engine):
    # create a session
    return Session(engine)


def get_all(model, session):
    # get all the records from the model
    return session.exec(select(model)).all()


def get_by_id(model, session, id):
    return session.get(model, id)


