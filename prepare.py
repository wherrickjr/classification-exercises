import pandas as pd
import os
import env
from sklearn.model_selection import train_test_split


def prep_iris(df):
    df = df.drop_duplicates()
    df.drop(columns = ['species_id', 'measurement_id'], inplace = True)
    df.rename(columns = {'species_name':'species'}, inplace = True)
    dummy_df = pd.get_dummies(df[['species']], dummy_na=False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_titanic(df):
    df = df.drop_duplicates()

    df = df.dropna()
    df.drop(columns = ['Unnamed: 0', 'passenger_id'\
        , 'survived', 'pclass', 'sibsp', 'parch', 'alone', 'embarked'],\
             inplace = True)
    list_of_columns = ['sex', 'class', 'deck', 'embark_town']
    dummy_df = pd.get_dummies(df[list_of_columns], dummy_na=False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

def prep_telco(df):
    df = df.drop_duplicates()
    df.drop(columns = ['Unnamed: 0', 'internet_service_type_id',\
                          'contract_type_id',\
                          'payment_type_id',\
                          ], inplace = True)
    list_of_columns = ['gender', 'partner',\
         'dependents', 'phone_service', 'multiple_lines','online_security',\
             'online_backup', 'device_protection', 'tech_support',\
                'streaming_movies', 'paperless_billing', 'churn', 'internet_service_type',\
                    'contract_type', 'payment_type', 'streaming_tv']
    dummy_df = pd.get_dummies(df[list_of_columns], dummy_na=False, drop_first=[True])
    df = pd.concat([df, dummy_df], axis=1)
    return df

def iris_split(df):
    train, test = train_test_split(df,test_size=.1,random_state=123, stratify=df.species)
    train, validate = train_test_split(train,test_size=.3,random_state=123, \
        stratify=train.species)

def titanic_split(df):
    train, test = train_test_split(df,test_size=.1,random_state=123, stratify=df.embark_town)
    train, validate = train_test_split(train,test_size=.3,random_state=123, \
        stratify=train.embark_town)

def telco_split(df):
    train, test = train_test_split(df,test_size=.1,random_state=123, stratify=df.churn)
    train, validate = train_test_split(train,test_size=.3,random_state=123, \
        stratify=train.churn)

def train_validate_test_split(df, col):
    train, test = train_test_split(df,test_size=.1,random_state=123, stratify= col)
    train, validate = train_test_split(train,test_size=.3,random_state=123, \
        stratify= col)
 