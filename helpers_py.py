import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def head_tail(df):
    print('First 5 rows of the DataFrame')
    print('\n')
    display(df.head())
    print('\n')
    print('Last 5 rows of the DataFrame')
    print('\n')
    display(df.head())

def miss_df(df):
    miss_df = pd.DataFrame()
    miss_df['count'] = df.isnull().sum()
    miss_df['percentage'] = round(df.isnull().sum() / len(df) * 100, 2)
    
    return miss_df

def miss_rows(df):
    miss_rows = df[df.isnull().any(axis = 1)]
    return miss_rows

def df_summary(df):
    print('No. of Samples      : ', df.shape[0])
    print('No. of Features     : ', df.shape[1])
    print('_____________________________________')
    print("\n Feature Names    : \n\n", df.columns.tolist())
    print('______________________________________')
    print("\n Categorical Features     : \n\n", df.select_dtypes(exclude = ['int', 'float']).columns.tolist())
    print('______________________________________')
    print("\n Numerical Features     : \n\n", df.select_dtypes(include = ['int', 'float']).columns.tolist())
    print('______________________________________')
    print("\n Missing Values   : \n\n", df.isnull().sum().values.sum())
    print("\n Missing Values Info   : \n\n", miss_df(df))
    print('______________________________________')
    print("\n Unique Values    : \n\n", df.nunique())
    print('______________________________________')
    print("\n Feature Type     : \n")
    print(df.info())
    print('______________________________________')
    try: 
    	print("\n Numerical Feature Descriptive Stats  : \n")
    	display(round(df.describe(include=['int', 'float']),3)) 
    except ValueError:
    	print("No Continuous Variables present in DataFrame")

    print('______________________________________')
    try: 
    	print("\n Categorical Feature Descriptive Stats  : \n")
    	display(round(df.describe(exclude=['int', 'float']),3)) 
    except ValueError:
    	print("No Categorical Variables present in DataFrame")

def categories(df):
    cat_features = df.select_dtypes(exclude = ['int', 'float']).columns.tolist()
    
    for col in cat_features:
        print(col)
        print(df[col].unique())
        print()

def corr_heatmap(df, size):

    sns.set_style("darkgrid")

    f, ax = plt.subplots(figsize=(size, size))

    mask = np.zeros(df.corr().shape, dtype=bool)

    mask[np.triu_indices(len(mask))] = True

    sns.heatmap(round(df.corr(),3), 
            vmin = -1, vmax = 1, center = 0, 
            cmap = "YlGnBu", 
            #cmap = sns.diverging_palette(220,10,as_cmap = True),
            annot = True,
            annot_kws={"size":size*0.6},
            mask = mask)

    ax.set_ylim(ax.get_ylim()[0]+0.5, ax.get_ylim()[1]-0.5)

def numeric_features(df):
	numeric_features = df.select_dtypes(include = ['int', 'float']).columns.tolist()
	return numeric_features

def cat_features(df):
	cat_features = df.select_dtypes(exclude = ['int', 'float']).columns.tolist()
	return cat_features

def cat_counts(df):
    cats = list(df.select_dtypes('object').columns)
    print(cats)
    print()

    for cat in cats:
        print(df[cat].value_counts())
        print('--------------------')

def class_distribution(df, col):
    print(df[col].value_counts())
    print()
    print((df[col].value_counts() / len(df) * 100).round(3))