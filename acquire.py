import pandas as pd
import os
import env
from sklearn.model_selection import train_test_split

def get_connection(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
    filename = 'titanic.csv'

    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))
        df.to_csv(filename)
        return df

def get_iris_data():
    filename = 'iris.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('select measurements.*, species.species_id, \
        species.species_name from species join \
            measurements using(species_id);', get_connection('iris_db'))
        df.to_csv(filename)
        return df

def get_telco_data():
    filename = 'telco.csv'
    if os.path.isfile(filename):
        return pd.read_csv(filename)
    else:
        df = pd.read_sql('select customers.*, i.internet_service_type, \
        c.contract_type, p.payment_type from customers\
            join internet_service_types as i using(internet_service_type_id) \
                join contract_types as c using(contract_type_id)\
                     join payment_types as p using(payment_type_id);', \
                        get_connection('telco_churn'))
        df.to_csv(filename)
        return df




