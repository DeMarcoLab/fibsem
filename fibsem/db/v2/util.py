from sqlalchemy import create_engine, Column, Integer, String, DateTime, ForeignKey, JSON, Boolean, Float
from sqlalchemy.orm import sessionmaker, declarative_base

from datetime import datetime
from fibsem import config as cfg
from pprint import pprint

# TODO: generate from config file

# # Load the YAML file
# with open('models.yaml', 'r') as file:
#     models = yaml.safe_load(file)

# # Generate the SQLAlchemy models
# for model_name, fields in models.items():
#     fields = {field_name: eval(field_type) for field_name, field_type in fields.items()}
#     model = type(model_name, (Base,), fields)

# # Create an engine that connects to a SQLite database
# engine = create_engine('sqlite:///my_database.db')

# # Create all tables in the engine
# Base.metadata.create_all(engine)

# Define the base class for declarative models
Base = declarative_base()

# Define the User model
class User(Base):
    __tablename__ = 'users'

    id = Column(Integer, primary_key=True)
    username = Column(String)
    name = Column(String)
    email = Column(String)
    password = Column(String)
    role = Column(String)
    default = Column(Boolean)
    preferences = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))

class Sample(Base):
    __tablename__ = 'samples'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    organism = Column(String)
    information = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))

class Instrument(Base):
    __tablename__ = 'instruments'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    serial_number = Column(String)
    manufacturer = Column(String)
    model = Column(String)
    description = Column(String)
    information = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))

class Configuration(Base):
    __tablename__ = 'configurations'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    path = Column(String)
    default = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))
    instrument_id = Column(Integer, ForeignKey('instruments.id'))


class Experiment(Base):
    __tablename__ = 'experiments'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    path = Column(String)
    method = Column(String)
    description = Column(String)
    information = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))
    project_id = Column(Integer, ForeignKey('projects.id'))
    sample_id = Column(Integer, ForeignKey('samples.id'))
    instrument_id = Column(Integer, ForeignKey('instruments.id'))

class Protocol(Base):
    __tablename__ = 'protocols'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    path = Column(String)
    default = Column(Boolean)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))

class Lamella(Base):
    __tablename__ = 'lamellae'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    state = Column(String)
    failed = Column(Boolean)
    description = Column(String)
    information = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))

class Step(Base):
    __tablename__ = 'steps'

    id = Column(Integer, primary_key=True)
    stage = Column(String)
    step = Column(String)
    step_n =  Column(Integer)
    timestamp = Column(DateTime)
    start_timestamp = Column(DateTime)
    end_timestamp = Column(DateTime)
    duration = Column(Integer)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    lamella_id = Column(Integer, ForeignKey('lamellae.id'))

class Detection(Base):
    __tablename__ = 'detections'

    id = Column(Integer, primary_key=True)
    lamella_id = Column(Integer, ForeignKey('lamellae.id'))
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    stage = Column(String)
    step = Column(String)
    feature = Column(String)
    is_correct = Column(Boolean)
    timestamp = Column(DateTime)
    filename = Column(String)
    beam_type = Column(String)
    px_x = Column(Integer)
    px_y = Column(Integer)
    dpx_x = Column(Integer)
    dpx_y = Column(Integer)
    dm_x = Column(Float)
    dm_y = Column(Float)   
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class Interaction(Base):
    __tablename__ = 'interactions'

    id = Column(Integer, primary_key=True)
    lamella_id = Column(Integer, ForeignKey('lamellae.id'))
    experiment_id = Column(Integer, ForeignKey('experiments.id'))
    stage = Column(String)
    step = Column(String)
    type = Column(String)
    subtype = Column(String)
    beam_type = Column(String)
    dm_x = Column(Float)
    dm_y = Column(Float)
    timestamp = Column(DateTime)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class Position(Base):
    __tablename__ = 'positions'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    x = Column(Float)
    y = Column(Float)
    z = Column(Float)
    r = Column(Float)
    t = Column(Float)
    coordinate_system = Column(String)
    instrument_id = Column(Integer, ForeignKey('instruments.id'))
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)



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

def create_user(session, user: User):
    session.add(user)
    session.commit()
    return user

def create_project(session, project: Project):
    session.add(project)
    session.commit()
    return project

def create_sample(session, sample: Sample):
    session.add(sample)
    session.commit()
    return sample

def create_instrument(session, instrument: Instrument):
    session.add(instrument)
    session.commit()
    return instrument


def create_experiment(session, experiment: Experiment):
    session.add(experiment)
    session.commit()
    return experiment

def create_lamella(session, lamella: Lamella):
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

def get_users(session):
    """Query the User table"""
    users = session.query(User).all()
    session.close()
    return users
