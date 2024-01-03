import numpy as np
import pandas as pd
import matplotlib as plt

def add_image(df, fruit):
    df_fruit_image = pd.DataFrame(fruit)
    df = pd.concat([df, df_fruit_image], axis = 1)
    df = df[df['recognized'] == True]
    df = df.drop(['recognized'], axis = 1)
    df = df.sample(n = 5000, replace=True)
    return df

def get_data():
    df_lion_original = pd.read_csv('../data/lion.csv')
    lion = np.load('../data_npy/lion.npy')
    df_lion = add_image(df_lion_original, lion)

    df_panda_original = pd.read_csv('../data/panda.csv')
    panda = np.load('../data_npy/panda.npy')
    df_panda = add_image(df_panda_original, panda)

    df_monkey_original = pd.read_csv('../data/monkey.csv')
    monkey = np.load('../data_npy/monkey.npy')
    df_monkey = add_image(df_monkey_original, monkey)

    df_duck_original = pd.read_csv('../data/duck.csv')
    duck = np.load('../data_npy/duck.npy')
    df_duck = add_image(df_duck_original, duck)
    return df_lion, df_panda, df_monkey, df_duck