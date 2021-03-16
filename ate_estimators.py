import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split

def ipw_ate(df, propensity_scores, t_col='is_weekend', y_col='shares'):
    return (((df[t_col] == 1).astype('float64') * df[y_col]) / propensity_scores \
            - ((df[t_col] == 0).astype('float64') * df[y_col]) \
            / (1 - propensity_scores)).mean()

def _distances(x0, x1):
    m = len(x0)
    n = len(x1)
    distances = np.zeros((n,m))

    for i in range(n):
        distances[i,:] = ((x1.iloc[i] - x0)**2).sum(axis=1)

    return distances

def matching_ate(df, t_col='is_weekend', y_col='shares'):
    x0 = df[df[t_col]==0].drop([t_col,y_col],axis=1)
    x1 = df[df[t_col]==1].drop([t_col,y_col],axis=1)
    y0 = df[df[t_col]==0][y_col]
    y1 = df[df[t_col]==1][y_col]
    d = _distances(x0, x1)
    return ((y1.values - y0.iloc[d.argmin(axis=1)].values).sum() \
        + (y1.iloc[d.T.argmin(axis=1)].values - y0.values).sum()) /  len(df)

def s_learner_ate(df, t_col='is_weekend', y_col='shares', **kwargs):
    xt = df.drop([y_col], axis=1)
    x1 = xt[xt[t_col]==1]
    y = df[y_col]
    
    f_x_t = GradientBoostingRegressor(**kwargs).fit(xt, y)
    
    xt_0 = xt.copy()
    xt_0[t_col] = 0
    xt_1 = xt.copy()
    xt_1[t_col] = 1
    
    return (f_x_t.predict(xt_1) - f_x_t.predict(xt_0)).mean()

def t_learner_ate(df, t_col='is_weekend', y_col='shares', **kwargs):
    x = df.drop([t_col,y_col], axis=1)
    x0 = df[df[t_col]==0].drop([t_col,y_col], axis=1)
    x1 = df[df[t_col]==1].drop([t_col,y_col], axis=1)
    y0 = df[df[t_col]==0][y_col]
    y1 = df[df[t_col]==1][y_col]
    
    f_x_0 = GradientBoostingRegressor(**kwargs).fit(x0, y0)
    f_x_1 = GradientBoostingRegressor(**kwargs).fit(x1, y1)
    
    print(f_x_0.score(x0, y0))
    print(f_x_1.score(x1, y1))
    
    return (f_x_1.predict(x) - f_x_0.predict(x)).mean()

def x_learner_ate(df, propensity_scores, t_col='is_weekend', y_col='shares', **kwargs):
    x = df.drop([t_col,y_col], axis=1)
    x0 = df[df[t_col]==0].drop([t_col,y_col],axis=1)
    x1 = df[df[t_col]==1].drop([t_col,y_col],axis=1)
    y0 = df[df[t_col]==0][y_col]
    y1 = df[df[t_col]==1][y_col]

    f_x_1 = GradientBoostingRegressor(**kwargs).fit(x1, y1)
    f_x_0 = GradientBoostingRegressor(**kwargs).fit(x0, y0)
    
    ite_1 = y1 - f_x_0.predict(x1)
    ite_0 = f_x_1.predict(x0) - y0
    
    f_ite_1 = GradientBoostingRegressor(**kwargs).fit(x1, ite_1)
    f_ite_0 = GradientBoostingRegressor(**kwargs).fit(x0, ite_0)
    
    return (propensity_scores * f_ite_0.predict(x) + (1 - propensity_scores) * f_ite_1.predict(x)).mean()