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
    else:
        raise NotImplementedError(method)
    propensity_model.fit(x, t)
    return lambda x: propensity_model.predict_proba(x)[:,1]

def trim_common_support(data, label_name):
    """ Removes observations that fall outside the common support of the propensity score 
        distribution from the data.
    
    Arguments:
    ----------
        data:        DataFrame with the propensity scores for each observation.
        label_name:  Column name that contains the labels (treatment/control) for each observation.
    
    """
    min_propensity = (data.groupby(label_name) \
                         .propensity.agg(min_propensity='min')).reset_index()['min_propensity']
    max_propensity = (data.groupby(label_name) \
                         .propensity.agg(max_propensity='max')).reset_index()['max_propensity']

    # Compute boundaries of common support between the two propensity score distributions
    min_common_support = np.max(min_propensity)
    max_common_support = np.min(max_propensity)

    common_support = (data.propensity >= min_common_support) & (data.propensity <= max_common_support)
    control = (data[label_name] == 0)
    treated = (data[label_name] == 1)
    
    return data[common_support]
