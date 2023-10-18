import sqlite3
from sqlite3 import Error

def create_table(conn, create_table_sql):
    """ create a table from the create_table_sql statement
    :param conn: Connection object
    :param create_table_sql: a CREATE TABLE statement
    :return:
    """
    try:
        c = conn.cursor()
        c.execute(create_table_sql)
    except Error as e:
        print(e)




def create_project(conn, project):
    """
    Create a new project into the projects table
    :param conn:
    :param project:
    :return: project id
    """
    sql = ''' INSERT INTO projects(name,date,user)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, project)
    conn.commit()
    return cur.lastrowid


def create_user(conn, user):
    """
    Create a new user into the users table
    :param conn:
    :param user:
    :return: user id
    """
    sql = ''' INSERT INTO users(name,email,password)
              VALUES(?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, user)
    conn.commit()
    return cur.lastrowid

def create_sample(conn, sample):
    """
    Create a new sample into the samples table
    :param conn:
    :param sample:
    :return: sample id
    """
    sql = ''' INSERT INTO samples(name,project_id,date,user)
              VALUES(?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, sample)
    conn.commit()
    return cur.lastrowid

def create_experiment(conn, experiment):
    """
    Create a new experiment into the experiments table
    :param conn:
    :param experiment:
    :return: experiment id
    """
    sql = ''' INSERT INTO experiments(name,project_id,date,user,sample_id,program,method,path)
              VALUES(?,?,?,?,?,?,?,?) '''
    cur = conn.cursor()
    cur.execute(sql, experiment)
    conn.commit()
    return cur.lastrowid

def create_connection(db_file):
    """ create a database connection to the SQLite database
        specified by the db_file
    :param db_file: database file
    :return: Connection object or None
    """
    conn = None
    try:
        conn = sqlite3.connect(db_file, 
        detect_types=sqlite3.PARSE_DECLTYPES |
                             sqlite3.PARSE_COLNAMES)
    except Error as e:
        print(e)

    return conn


#####
SQL_CREATE_PROJECTS_TABLE = """ CREATE TABLE IF NOT EXISTS projects (
                                    id INTEGER PRIMARY KEY,
                                    name VARCHAR(100) NOT NULL,
                                    date TIMESTAMP NOT NULL,
                                    user VARCHAR(100) NOT NULL
                                ); """

SQL_CREATE_USERS_TABLES = """CREATE TABLE IF NOT EXISTS users (
                                id INTEGER PRIMARY KEY,
                                name VARCHAR(100) NOT NULL,
                                email VARCHAR(100) NOT NULL,
                                password VARCHAR(100) NOT NULL
                            );"""


SQL_CREATE_SAMPLES_TABLE = """CREATE TABLE IF NOT EXISTS samples (
                                id INTEGER PRIMARY KEY,
                                name VARCHAR(100) NOT NULL,
                                project_id INTEGER NOT NULL,
                                date TIMESTAMP NOT NULL,
                                user VARCHAR(100) NOT NULL,
                                FOREIGN KEY (project_id) REFERENCES projects (id)
                                );"""


SQL_CREATE_EXPERIMENTS_TABLE = """CREATE TABLE IF NOT EXISTS experiments (
                                id INTEGER PRIMARY KEY,
                                name VARCHAR(100) NOT NULL,
                                project_id INTEGER NOT NULL,
                                date TIMESTAMP NOT NULL,
                                user VARCHAR(100) NOT NULL,
                                sample_id INTEGER NOT NULL,
                                program VARCHAR(100) NOT NULL,
                                method VARCHAR(100) NOT NULL,
                                path VARCHAR(100) NOT NULL,
                                FOREIGN KEY (project_id) REFERENCES projects (id)
                                FOREIGN KEY (sample_id) REFERENCES samples (id)
                            );"""



SQL_CREATE_HISTORY_TABLE = """CREATE TABLE IF NOT EXISTS history (
                                id INTEGER PRIMARY KEY,
                                petname VARCHAR(100) NOT NULL,
                                stage VARCHAR(100) NOT NULL,
                                start TIMESTAMP NOT NULL,
                                end TIMESTAMP NOT NULL,
                                duration FLOAT NOT NULL,
                                experiment_id INTEGER NOT NULL,
                                FOREIGN KEY (experiment_id) REFERENCES experiment (id)
                                
                            );"""

SQL_CREATE_STEPS_TABLE = """CREATE TABLE IF NOT EXISTS steps (
                                id INTEGER PRIMARY KEY,
                                petname VARCHAR(100) NOT NULL,
                                stage VARCHAR(100) NOT NULL,
                                step VARCHAR(100) NOT NULL,
                                step_n INTEGER NOT NULL,
                                timestamp TIMESTAMP NOT NULL,
                                duration FLOAT NOT NULL,
                                experiment_id INTEGER NOT NULL,
                                FOREIGN KEY (experiment_id) REFERENCES experiment (id)
                                
                            );"""



SQL_CREATE_DETECTIONS_TABLE = """CREATE TABLE IF NOT EXISTS detections (
                                id INTEGER PRIMARY KEY,
                                petname VARCHAR(100) NOT NULL,
                                stage VARCHAR(100) NOT NULL,
                                step VARCHAR(100) NOT NULL,
                                feature VARCHAR(100) NOT NULL,
                                px_x INTEGER NOT NULL,
                                px_y INTEGER NOT NULL,
                                dpx_x INTEGER NOT NULL,
                                dpx_y INTEGER NOT NULL,
                                dm_x FLOAT NOT NULL,
                                dm_y FLOAT NOT NULL,
                                is_correct BOOL NOT NULL,
                                beam_type VARCHAR(100) NOT NULL,
                                fname VARCHAR(100) NOT NULL,
                                timestamp TIMESTAMP NOT NULL,
                                experiment_id INTEGER NOT NULL,
                                FOREIGN KEY (experiment_id) REFERENCES experiment (id)
                                
                            );"""


SQL_CREATE_INTERACTIONS_TABLE = """CREATE TABLE IF NOT EXISTS interactions (
                                id INTEGER PRIMARY KEY,
                                petname VARCHAR(100) NOT NULL,
                                stage VARCHAR(100) NOT NULL,
                                step VARCHAR(100) NOT NULL,
                                type VARCHAR(100) NOT NULL,
                                subtype VARCHAR(100) NOT NULL, 
                                dm_x FLOAT NOT NULL,
                                dm_y FLOAT NOT NULL,
                                beam_type VARCHAR(100) NOT NULL,
                                timestamp TIMESTAMP NOT NULL,
                                experiment_id INTEGER NOT NULL,
                                FOREIGN KEY (experiment_id) REFERENCES experiment (id)
                                
                            );"""

from fibsem import config as cfg
import os

def _create_database():

    # create / connect to db
    database = cfg.DATABASE_PATH
    conn = create_connection(database)

    # create tables
    create_table(conn, SQL_CREATE_PROJECTS_TABLE)
    create_table(conn, SQL_CREATE_USERS_TABLES)
    create_table(conn, SQL_CREATE_SAMPLES_TABLE)
    create_table(conn, SQL_CREATE_EXPERIMENTS_TABLE)
    create_table(conn, SQL_CREATE_HISTORY_TABLE)
    create_table(conn, SQL_CREATE_STEPS_TABLE)
    create_table(conn, SQL_CREATE_DETECTIONS_TABLE)
    create_table(conn, SQL_CREATE_INTERACTIONS_TABLE)