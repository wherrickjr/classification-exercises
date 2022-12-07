import pandas as pd

import env

def get_connection(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

def get_iris_data():
    return pd.read_sql('select measurements.*, species.species_id, \
        species.species_name from species join \
            measurements using(species_id);', get_connection('iris_db'))