import pandas as pd
from prep import get_ratio
from datetime import datetime, timedelta

stime, etime = 'formatted_start_time', 'formatted_end_time'
dformat = '%Y-%m-%d %H:%M:%S'

    
def get_between_with_duration(df0, target, merge_way, s, e, duration=None):
    df = df0.copy()
    df[stime] = pd.to_datetime(df[stime])
    df[etime] = pd.to_datetime(df[etime])
    df_avg = None
    if duration is None:
        duration = 'manual_duration'
        df[duration] = (df[etime] - df[stime]).apply(lambda x: x.total_seconds())
    df = df[['productName', 'productVersion', 'timestamps', stime, etime, 'enSN', target, duration]]
    df_avg = pd.DataFrame(columns=df.columns)
    df1 = df[(df[etime] >= s) & (df[stime] < e)].reset_index(drop=True).copy()
    if len(df1) == 0:
        df1 = pd.DataFrame({'productName': df0['productName'].iat[0], 'productVersion': df0['productVersion'].iat[0],
                            'timestamps': df0['timestamps'].iat[0], 'enSN': df0['enSN'].iat[0],
                            target: 0.0, duration: 0.0}, index=[0])
    else:
        ratios = df1.apply(lambda r: get_ratio(r[stime], r[etime], s, e), axis=1)
        df1[duration] *= ratios
        if merge_way == 'mean':
            df1[target] = df1[target] * df1[duration]
            df1 = df1.groupby(['productName', 'productVersion', 'timestamps', 'enSN'])[target, duration].sum().reset_index()
            df1[target] = df1[target] / df1[duration]
        if merge_way == 'sum':
            df1[target] *= ratios
            df1 = df1.groupby(['productName', 'productVersion', 'timestamps', 'enSN'])[target, duration].sum().reset_index()
    df1[stime], df1[etime]= s, e
    df_avg = df_avg.append(df1, ignore_index=True)
    df_avg[target] = pd.to_numeric(df_avg[target])
    #df_avg.drop(columns=[duration], inplace=True)
    df_avg[duration] = pd.to_numeric(df_avg[duration])    
    #df_avg[stime] = df_avg[stime].apply(lambda s: datetime.strftime(s, dformat))
    #df_avg[etime] = df_avg[etime].apply(lambda s: datetime.strftime(s, dformat))
    return df_avg[target].iat[0], df_avg[duration].iat[0]


def get_between(df0, target, merge_way, s, e, duration=None):
    targ, _ = get_between_with_duration(df0, target, merge_way, s, e, duration)
    return targ
    
    
def get_point(df0, target, t, default_value=0):
    df = df0.copy()
    df[stime] = pd.to_datetime(df[stime])
    df[etime] = pd.to_datetime(df[etime])
    df1 = df[(df[etime] >= t) & (df[stime] <= t)]
    if len(df1) == 0:
        return default_value
    else:
        return df1[target].iat[0]


def get_mode(df, target, t, default_value=0):
    d = {1: [1, 0, 0, 0], 2: [0, 1, 0, 0], 4: [0, 0, 1, 0], 5: [0, 0, 0, 1]}
    return d[int(get_point(df, target, t, default_value))]


def get_bright_session_num(df, t, duration=10):
    t_before = t - timedelta(minutes=duration)
    return len(df[(df['last_bright_start_time'] <= t) & (df['last_bright_start_time'] >= t_before)])


def get_apps_between_with_duration(df, target, merge_way, s, e, apps, duration=None):
    if len(apps) == 0:
        return 0, 0
    df0 = df[df['name'].isin(apps)].copy()
    return get_between_with_duration(df0, target, merge_way, s, e, duration)
    
    
if __name__ == '__main__':
    # df = pd.read_csv('./data/db_brightness_detail.csv')
    # df_user = df[df['enSN'] == 'ELS000040'].reset_index(drop=True)
    # target, duration = 'brightness', 'duration'
    # s = datetime(2020, 5, 17, 9, 36, 19)
    # e = datetime(2020, 5, 17, 9, 38, 20)
    # df_user_norm = get_between(df_user, target, 'sum', s, e, duration)

    df1 = pd.read_csv('./data/db_ambient_light_detail.csv')
    df_user1 = df1[df1['enSN'] == 'ELS000040'].reset_index(drop=True)
    target = 'level'
    t = datetime(2020, 5, 17, 11, 26, 32)
    point = get_point(df_user1, target, t)
    
    