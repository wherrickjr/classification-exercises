import pandas as pd
import os
import env

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


def prep_iris(df):
    df = df.drop_duplicates()
    df.drop(columns = ['species_id', 'measurement_id'], inplace = True)
    df.rename(columns = {'species_name':'species'}, inplace = True)
    dummy_df = pd.get_dummies(df[['species']], dummy_na=False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

