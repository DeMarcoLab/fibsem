# SQLModel based models



from typing import Optional, Union

from sqlmodel import Field, SQLModel, create_engine, select
from sqlalchemy import Column, DateTime, func, JSON


from datetime import datetime


class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str
    name: str
    email: str
    password: str
    role: str
    additional_info: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    default: bool = Field(default=False)
    created_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now()
        ))
    updated_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now(),
        server_onupdate=func.now()
        ))
    
    class Config:
        arbitrary_types_allowed = True


class Project(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    additional_info: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now()
        ))
    updated_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now(),
        server_onupdate=func.now()
        ))
    user_id: int = Field(default=None, foreign_key="user.id")




class Sample(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    organism: str
    additional_info: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now()
        ))
    updated_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now(),
        server_onupdate=func.now()
        ))
    user_id: int = Field(default=None, foreign_key="user.id")


class Instrument(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    serial_number: str
    manufacturer: str
    model: str
    additional_info: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now()
        ))
    updated_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now(),
        server_onupdate=func.now()
        ))
    user_id: int = Field(default=None, foreign_key="user.id")


class Configuration(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    path: str
    default: bool = False
    additional_info: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now()
        ))
    updated_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now(),
        server_onupdate=func.now()
        ))
    user_id: int = Field(default=None, foreign_key="user.id")
    instrument_id: int = Field(default=None, foreign_key="instrument.id")


class Experiment(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    path: str
    method: str
    additional_info: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now()
        ))
    updated_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now(),
        server_onupdate=func.now()
        ))
    user_id: int = Field(default=None, foreign_key="user.id")
    project_id: int = Field(default=None, foreign_key="project.id")
    sample_id: int = Field(default=None, foreign_key="sample.id")
    instrument_id: int = Field(default=None, foreign_key="instrument.id")
    configuration_id: int = Field(default=None, foreign_key="configuration.id")

class Protocol(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    method: str
    path: str
    default: bool = False
    data: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now()
        ))
    updated_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now(),
        server_onupdate=func.now()
        ))
    user_id: int = Field(default=None, foreign_key="user.id")


class Lamella(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    description: str
    state: str
    failed: bool = Field(default=False)
    data: Optional[dict] = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now()
        ))
    updated_at: datetime = Field(sa_column=Column(
        DateTime(timezone=True),
        server_default=func.now(),
        server_onupdate=func.now()
        ))
    experiment_id: int = Field(default=None, foreign_key="experiment.id")