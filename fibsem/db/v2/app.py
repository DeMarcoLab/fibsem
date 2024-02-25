
from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from fibsem.db.v2.models import Base, User, Project, Sample, Instrument, Experiment, Lamella, Step, Detection, Interaction
from fibsem.db.v2 import util
from fibsem import config as cfg


app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# create a connection to the database
engine = util.create_connection()

# create a session
session = util.create_session(engine)

# return all users
@app.get("/users")
async def get_users():
    users = session.query(User).all()
    return users

@app.get("/users/{user_id}")
async def get_user(user_id: int):
    user = session.query(User).filter(User.id == user_id).first()
    return user

# return all projects
@app.get("/projects")
async def get_projects():
    projects = session.query(Project).all()
    return projects

@app.get("/projects/{project_id}")
async def get_project(project_id: int):
    project = session.query(Project).filter(Project.id == project_id).first()
    return project


# return all samples
@app.get("/samples")
async def get_samples():
    samples = session.query(Sample).all()
    return samples

@app.get("/samples/{sample_id}")
async def get_sample(sample_id: int):
    sample = session.query(Sample).filter(Sample.id == sample_id).first()
    return sample


@app.get("/experiments")
async def get_experiments():
    experiments = session.query(Experiment).all()

    # drop the .data attribute
    for experiment in experiments:
        experiment.data = None

    return experiments

@app.get("/instruments")
async def get_instruments():
    instruments = session.query(Instrument).all()
    return instruments

@app.post("/user")
async def create_user(user: dict):
    user = User.from_dict(user)
    user = await util.create_user(session, user)
    return {"status": "ok", "data": user}


@app.post("/project")
async def create_project(project: dict):
    project = Project.from_dict(project)
    project = await util.create_project(session, project)
    return {"status": "ok", "data": project}


@app.post("/sample")
async def create_sample(sample: dict):
    sample = Sample.from_dict(sample)
    sample = await util.create_sample(session, sample)
    return {"status": "ok", "data": sample}


@app.put("/sample/{sample_id}")
async def update_sample(sample_id: int, sample: dict):
    sample = Sample.from_dict(sample)
    sample = await util.update_sample(session, sample_id, sample)
    return {"status": "ok", "data": sample}


# https://fastapi.tiangolo.com/tutorial/sql-databases/#__tabbed_1_2