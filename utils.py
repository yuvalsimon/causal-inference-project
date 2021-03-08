import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

random_state = 101

def factorize(df):
    """
    converts all categorical columns to numerical columns
    """
    groups = df.columns.to_series().groupby(df.dtypes).groups
    if(np.dtype('O') in groups):
        for s in groups[np.dtype('O')]:
            df[s] = df[s].factorize()[0]

def propensity_func(df, t_col='is_weekend', y_col='shares', method='log', **kwargs):
    x = df.drop([t_col, y_col],axis=1)
    t = df[t_col]

    if(method == 'log'):
        propensity_model = LogisticRegression(random_state=random_state, **kwargs)
    elif(method == 'random_forest'):
        propensity_model = RandomForestClassifier(random_state=random_state, **kwargs)
    elif(method == 'boosting'):
        propensity_model = GradientBoostingClassifier(random_state=random_state, **kwargs)
    propensity_model.fit(x, t)
    return lambda x: propensity_model.predict_proba(x)[:,1]