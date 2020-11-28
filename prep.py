import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

stime, etime = 'formatted_start_time', 'formatted_end_time'
dformat = '%Y-%m-%d %H:%M:%S'

def get_all_times(date):
    all_t = []
    year, month, day = date.split('-')
    for h in range(24):
        for m in range(0, 60, 10):
            all_t.append(datetime(int(year), int(month), int(day), h, m, 0))
    all_t.append(datetime(int(year), int(month), int(day) + 1, 0, 0, 0))
    return all_t

def get_ratio(stime, etime, all_stime, all_etime):
    if stime == etime:
        return 1.0
    a, b = max(stime, all_stime), min(etime, all_etime)
    return (b - a) / (etime - stime)

def split_user(df0):
    df = df0.copy()
    df.sort_values(by=['enSN', stime], inplace=True)
    df_dict = {}
    for i in df['enSN'].unique():
        df_user = df[df['enSN'] == i].reset_index(drop=True)
        df_dict[i] = df_user
        # df_user.to_csv(i + '_brightness.csv')
    return df_dict

def normalize(df0, target, merge_way, duration=None):
    df = df0.copy()
    all_t = get_all_times(df['timestamps'].iat[0])
    df[stime] = pd.to_datetime(df[stime])
    df[etime] = pd.to_datetime(df[etime])
    df_avg = None
    if duration is None:
        duration = 'manual_duration'
        df[duration] = (df[etime] - df[stime]).apply(lambda x: x.total_seconds())
    df = df[['productName', 'productVersion', 'timestamps', stime, etime, 'enSN', target, duration]]
    df_avg = pd.DataFrame(columns=df.columns)
    for i in range(len(all_t) - 1):
        df1 = df[(df[etime] >= all_t[i]) & (df[stime] < all_t[i+1])].reset_index(drop=True).copy()
        if len(df1) == 0:
            df1 = pd.DataFrame({'productName': df0['productName'].iat[0], 'productVersion': df0['productVersion'].iat[0],
                                'timestamps': df0['timestamps'].iat[0], 'enSN': df0['enSN'].iat[0],
                                target: 0.0, duration: 0.0}, index=[0])
        else:
            ratios = df1.apply(lambda r: get_ratio(r[stime], r[etime], all_t[i], all_t[i+1]), axis=1)
            df1[duration] *= ratios
            if merge_way == 'mean':
                df1[target] = df1[target] * df1[duration]
                df1 = df1.groupby(['productName', 'productVersion', 'timestamps', 'enSN'])[target, duration].sum().reset_index()
                df1[target] = df1[target] / df1[duration]
            if merge_way == 'sum':
                df1[target] *= ratios
                df1 = df1.groupby(['productName', 'productVersion', 'timestamps', 'enSN'])[target, duration].sum().reset_index()
        df1[stime], df1[etime]= all_t[i], all_t[i+1]
        df_avg = df_avg.append(df1, ignore_index=True)
    df_avg[target] = pd.to_numeric(df_avg[target])
    df_avg.drop(columns=[duration], inplace=True)
    #df_avg[duration] = pd.to_numeric(df_avg[duration])    
    df_avg[stime] = df_avg[stime].apply(lambda s: datetime.strftime(s, dformat))
    df_avg[etime] = df_avg[etime].apply(lambda s: datetime.strftime(s, dformat))
    return df_avg

def prep_plot(df0, col):
    df = df0.copy()
    year, month, day = df['timestamps'].iat[0].split('-')
    t_array = [datetime(int(year), int(month), int(day), 0, 0, 0), datetime.strptime(df[stime][0], dformat)]
    b_array = [0, 0]
    for index, row in df.iterrows():
        if datetime.strptime(row[stime], dformat) == t_array[-1]:
            t_array.append(datetime.strptime(row[stime], dformat))
            t_array.append(datetime.strptime(row[etime], dformat))
            b_array.extend([row[col]] * 2)
        else:
            t_array.append(t_array[-1])
            t_array.append(datetime.strptime(row[stime], dformat))
            b_array.extend([0, 0])            
            t_array.append(datetime.strptime(row[stime], dformat))
            t_array.append(datetime.strptime(row[etime], dformat))
            b_array.extend([row[col]] * 2)
    if t_array[-1] != datetime(int(year), int(month), int(day) + 1, 0, 0, 0):
        t_array.append(t_array[-1])
        t_array.append(datetime(int(year), int(month), int(day) + 1, 0, 0, 0))
        b_array.extend([0, 0])
    return t_array, b_array

def plot(df0, col):
    t_array, b_array = prep_plot(df0, col)
    plt.figure()
    plt.plot(t_array, b_array)
    plt.title(df0['enSN'][0])
    plt.xlabel('Datetime')
    plt.ylabel(col)
    plt.show()

if __name__ == '__main__':
    df = pd.read_csv('./data/db_brightness_detail.csv')
    df_dict = split_user(df)
    df_user = df_dict['ELS000040']
    target, duration = 'brightness', 'duration'
    plot(df_user, target)
    #df_user.to_csv('ELS000040_brightness.csv', index=False, encoding='utf-8-sig')
    df_user_norm = normalize(df_user, target, 'mean', duration)
    plot(df_user_norm, target)
    #df_user_norm.to_csv('ELS000040_brightness_norm.csv', index=False, encoding='utf-8-sig')
    df_user_norm_sum = normalize(df_user, target, 'sum')
    plot(df_user_norm_sum, target)
    