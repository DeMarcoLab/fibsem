
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from fibsem import config as cfg
from fibsem.db.v2.models import Base, User, Project, Sample, Instrument, Experiment, Lamella, Step, Detection, Interaction

def create_connection(database_path: str = cfg.DATABASE_PATH):
    # create a connection to the database
    engine = create_engine(f'sqlite:///{database_path}')

    return engine

def create_database(engine):
    # create the database
    Base.metadata.create_all(engine)
    
def create_session(engine):
    # create a session
    Session = sessionmaker(bind=engine)
    session = Session()
    return session

async def create_user(session, user: User):
    session.add(user)
    session.commit()
    return user

async def create_project(session, project: Project):
    session.add(project)
    session.commit()
    return project

async def create_sample(session, sample: Sample):
    session.add(sample)
    session.commit()
    return sample

async def create_instrument(session, instrument: Instrument):
    session.add(instrument)
    session.commit()
    return instrument


async def create_experiment(session, experiment: Experiment):
    session.add(experiment)
    session.commit()
    return experiment

async def create_lamella(session, lamella: Lamella):
    session.add(lamella)
    session.commit()
    return lamella

def create_step(session, step: Step):
    session.add(step)
    session.commit()
    return step

def create_detection(session, detection: Detection):
    session.add(detection)
    session.commit()
    return detection

def create_interaction(session, interaction: Interaction):
    session.add(interaction)
    session.commit()
    return interaction


def add_steps(session, steps: list):
    session.add_all(steps)
    session.commit()
    return steps

def add_detections(session, detections: list):
    session.add_all(detections)
    session.commit()
    return detections

def add_interactions(session, interactions: list):
    session.add_all(interactions)
    session.commit()
    return interactions



def get_user(session, username: str):
    user = session.query(User).filter(User.username == username).first()
    session.close()
    return user

def get_project(session, name: str):
    project = session.query(Project).filter(Project.name == name).first()
    session.close()
    return project

def get_sample(session, name: str):
    sample = session.query(Sample).filter(Sample.name == name).first()
    session.close()
    return sample

def get_instrument(session, name: str):
    instrument = session.query(Instrument).filter(Instrument.name == name).first()
    session.close()
    return instrument

def get_projects(session):
    projects = session.query(Project).all()
    session.close()
    return projects

def get_samples(session):
    samples = session.query(Sample).all()
    session.close()
    return samples

def get_instruments(session):
    instruments = session.query(Instrument).all()
    session.close()
    return instruments

def get_experiments(session):
    experiments = session.query(Experiment).all()
    session.close()
    return experiments

def get_users(session):
    """Query the User table"""
    users = session.query(User).all()
    session.close()
    return users
