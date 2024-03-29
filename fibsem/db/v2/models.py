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
    default = Column(Boolean, default=False)
    preferences = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def __repr__(self):
        return f'User(id={self.id}, username={self.username}, name={self.name}, email={self.email}, role={self.role}, default={self.default}, preferences={self.preferences}, created_at={self.created_at}, updated_at={self.updated_at})'
    
    @classmethod
    def from_dict(cls, user: dict):
        return cls(username=user['username'], name=user['name'], email=user['email'], 
                   password=user['password'], role=user['role'], 
                   default=user.get('default', False), preferences=user.get('preferences', {}))

class Project(Base):
    __tablename__ = 'projects'

    id = Column(Integer, primary_key=True)
    name = Column(String)
    description = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    user_id = Column(Integer, ForeignKey('users.id'))

    @classmethod
    def from_dict(cls, project: dict):
        return cls(name=project['name'], 
                   description=project['description'], 
                   user_id=project['user_id'])

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
    data = Column(JSON)
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
    data = Column(JSON)
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

