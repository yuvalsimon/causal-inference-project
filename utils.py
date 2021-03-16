from collections import namedtuple
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import RFE
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

prop_func = namedtuple("prop_func", "func model")

def propensity_func(df, t_col='is_weekend', y_col='shares', method='log', **kwargs):
    x = df.drop([t_col, y_col],axis=1)
    t = df[t_col]

    if(method == 'log'):
        propensity_model = LogisticRegression(random_state=random_state, **kwargs)
#         propensity_model = RFE(estimator=propensity_model)
    elif(method == 'random_forest'):
        propensity_model = RandomForestClassifier(random_state=random_state, **kwargs)
#         propensity_model = RFE(estimator=propensity_model)
    elif(method == 'boosting'):
        propensity_model = GradientBoostingClassifier(random_state=random_state, **kwargs)
#         propensity_model = RFE(estimator=propensity_model)
    else:
        raise NotImplementedError(method)
    propensity_model.fit(x, t)
    return prop_func(lambda x: propensity_model.predict_proba(x)[:,1], propensity_model)

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