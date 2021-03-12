import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
import holidays

random_state = 101

us_holidays = holidays.UnitedStates(years=[2013, 2014])

def is_holiday(date):
    return date in us_holidays

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

# def trim_common_support(data, label_name):
#     """ Removes observations that fall outside the common support of the propensity score 
#         distribution from the data.
    
#     Arguments:
#     ----------
#         data:        DataFrame with the propensity scores for each observation.
#         label_name:  Column name that contains the labels (treatment/control) for each observation.
    
#     """
#     min_propensity = (data.groupby(label_name) \
#                          .propensity.agg(min_propensity='min')).reset_index()['min_propensity']
#     max_propensity = (data.groupby(label_name) \
#                          .propensity.agg(max_propensity='max')).reset_index()['max_propensity']

#     # Compute boundaries of common support between the two propensity score distributions
#     min_common_support = np.max(min_propensity)
#     max_common_support = np.min(max_propensity)

#     common_support = (data.propensity >= min_common_support) & (data.propensity <= max_common_support)
#     control = (data[label_name] == 0)
#     treated = (data[label_name] == 1)
    
#     return data[common_support]

def propensity_hist_values(df, t, t_label='is_weekend'):
    hist = (df[df[t_label] == t]['propensity'] * 100).apply(np.round).astype('int').value_counts()
    res = np.zeros(101)
    res[hist.index] = hist.values
    return res

def trim_common_support(df, t_label='is_weekend', min_articles_count=15):
    t0_hist = propensity_hist_values(df, 0, t_label)
    t1_hist = propensity_hist_values(df, 1, t_label)
    good_points = np.logical_and(t0_hist >= min_articles_count, t1_hist >= min_articles_count)
    largest_range_first_idx = -1
    largest_range_len = 0
    first_idx = 0
    cur_count = 0
    for i in range(len(good_points)):
        if(good_points[i]):
            cur_count += 1
        else:
            if(cur_count > largest_range_len):
                largest_range_first_idx = first_idx
                largest_range_len = cur_count
            cur_count = 0
            first_idx = i + 1
    min_propensity = largest_range_first_idx / 100
    max_propensity = (largest_range_first_idx + largest_range_len) / 100
    common_support = (df.propensity >= min_propensity) & (df.propensity <= max_propensity)
    return df[common_support], df[common_support == False]