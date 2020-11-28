import pandas as pd
import torch
import os
from features import get_between

stime, etime = 'formatted_start_time', 'formatted_end_time'
ltime = 'last_bright_start_time'

def load_user_brit(enSN):
    df = pd.read_csv('./data/db_brightness_detail.csv')
    return df[df['enSN'] == enSN].sort_values(by=[stime]).reset_index(drop=True).copy()

def load(filename, enSN):
    df = pd.read_csv(os.path.join('./data', filename))
    return df[df['enSN'] == enSN].reset_index(drop=True).copy()

def build_dark_session(enSN):
    df = load_user_brit(enSN)
    df_sess = pd.DataFrame()
    sess_s, sess_e, sess_flag = None, None, None
    for i, row in df.iterrows():
        if i == 0:
            sess_s, sess_e, sess_flag = row[stime], row[etime], row['brightness'] > 0
        else:
            curr_s, curr_e, curr_flag = row[stime], row[etime], row['brightness'] > 0
            if curr_s < sess_e:
                continue
            elif curr_s == sess_e:
                if curr_flag == sess_flag:
                    sess_e = curr_e
                else:
                    df_sess = df_sess.append({stime: sess_s, etime: sess_e, 'is_bright': sess_flag}, ignore_index=True)
                    sess_s, sess_e, sess_flag = curr_s, curr_e, curr_flag
            else:
                if not sess_flag:
                    if curr_flag == sess_flag:
                        sess_e = curr_e
                    else:
                        df_sess = df_sess.append({stime: sess_s, etime: curr_s, 'is_bright': sess_flag}, ignore_index=True)
                        sess_s, sess_e, sess_flag = curr_s, curr_e, curr_flag
                else:
                    df_sess = df_sess.append({stime: sess_s, etime: sess_e, 'is_bright': sess_flag}, ignore_index=True)
                    if not curr_flag:
                        sess_s, sess_e, sess_flag = sess_e, curr_e, curr_flag
                    else:
                        df_sess = df_sess.append({stime: sess_e, etime: curr_s, 'is_bright': False}, ignore_index=True)
                        sess_s, sess_e, sess_flag = curr_s, curr_e, curr_flag
    df_sess[ltime] = ''
    for i, row in df_sess.iterrows():
        if i > 0 and not row['is_bright']:
            df_sess[ltime].iat[i] = df_sess[stime][i-1]
    df_sess = df_sess[(df_sess['is_bright'] == False) & (df_sess[ltime] != '')]
    df_sess[stime] = pd.to_datetime(df_sess[stime])
    df_sess[etime] = pd.to_datetime(df_sess[etime])
    df_sess[ltime] = pd.to_datetime(df_sess[ltime])
    return df_sess.drop(columns=['is_bright']).reset_index(drop=True).copy()

def build_series(enSN, dt=pd.Timedelta(seconds=10)):
    df_sess = build_dark_session(enSN)
    df_wifi = load('db_app_wifi_data_detail.csv', enSN)
    df_cell = load('db_app_modem_data_detail.csv', enSN)
    df_audio = load('db_app_audio_detail.csv', enSN)
    df_disp = load('db_app_display_detail.csv', enSN)
    df_light = load('db_ambient_light_detail.csv', enSN)
    df_tp = load('db_app_tp_detail.csv', enSN)
    df_bt = load('db_app_bt_detail.csv', enSN)
    df_camera = load('db_app_camera_detail.csv', enSN)
    #df_cpu = load('db_app_cpu_detail.csv', enSN)
    df_gnss = load('db_app_gnss_detail.csv', enSN)
    df_gpu = load('db_app_gpu_detail.csv', enSN)
    df_sensor = load('db_app_sensor_detail.csv', enSN)
    series_dict = {}
    for i, row in df_sess.iterrows():
        series_dict[i] = pd.DataFrame(columns=['day of week', 'hour', 'minute', 'second', 'time before bright'])
        curr = row[stime]
        audio_energy = get_between(df_audio, 'screen_on_energy', 'sum', row[ltime], row[stime])
        disp_energy = get_between(df_disp, 'energy', 'sum', row[ltime], row[stime])
        light_level = get_between(df_light, 'level', 'mean', row[ltime], row[stime], duration='duration')
        tp_energy = get_between(df_tp, 'energy', 'sum', row[ltime], row[stime])
        bt_energy = get_between(df_bt, 'screen_on_energy', 'sum', row[ltime], row[stime])
        camera_energy = get_between(df_camera, 'screen_on_energy', 'sum', row[ltime], row[stime])
        #cpu_energy = get_between(df_cpu, 'screen_on_energy', 'sum', row[ltime], row[stime])
        gnss_energy = get_between(df_gnss, 'screen_on_energy', 'sum', row[ltime], row[stime])
        gpu_energy = get_between(df_gpu, 'energy', 'sum', row[ltime], row[stime])
        sensor_energy = get_between(df_sensor, 'screen_on_energy', 'sum', row[ltime], row[stime])
        while curr <= row[etime]:
            item = {'wifi data download since last session': get_between(df_wifi, 'screen_off_rx_bytes', 'sum', row[stime], curr),
                    'duration of last session': (row[stime] - row[ltime]).total_seconds(),
                    'time since last session': (curr - row[stime]).total_seconds(),
                    'wifi data upload since last session': get_between(df_wifi, 'screen_off_tx_bytes', 'sum', row[stime], curr),
                    'hour': curr.hour,
                    'day of week': curr.dayofweek,
                    'cellular data upload since last session': get_between(df_cell, 'screen_off_tx_bytes', 'sum', row[stime], curr),
                    'cellular data download since last session': get_between(df_cell, 'screen_off_rx_bytes', 'sum', row[stime], curr),
                    'minute': curr.minute,
                    'second': curr.second,
                    'audio energy during last session': audio_energy,
                    'display energy during last session': disp_energy,
                    'ambient light level during last session': light_level,
                    'tp energy during last session': tp_energy,
                    'bt energy during last session': bt_energy,
                    'camera energy during last session': camera_energy,
                    #'cpu energy during last session': cpu_energy,
                    'gnss energy during last session': gnss_energy,
                    'gpu energy during last session': gpu_energy,
                    'sensor energy during last session': sensor_energy,
                    'time before bright': (row[etime] - curr).total_seconds()}
            series_dict[i] = series_dict[i].append(item, ignore_index=True)
            curr += dt
    return series_dict

result = build_series('ELS000040')