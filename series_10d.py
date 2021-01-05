import pandas as pd
import numpy as np
import datetime
import os
from tqdm import tqdm
from features import get_between, get_point, get_mode, get_bright_session_num, get_apps_between_with_duration, get_app_num

stime, etime = 'formatted_start_time', 'formatted_end_time'
ltime = 'last_bright_start_time'

def load_user_brit(enSN):
    df = pd.read_csv('./data_10d/db_brightness_detail.csv')
    return df[df['enSN'] == enSN].sort_values(by=[stime]).reset_index(drop=True).copy()

def load(filename, enSN):
    df = pd.read_csv(os.path.join('./data_10d', filename))
    return df[df['enSN'] == enSN].reset_index(drop=True).copy()

def load_app_class(enSN, typ):
    df = pd.read_csv('./data/gpu_detail_type(1)(1).csv')
    return df['name'][(df['enSN'] == enSN) & (df['type'] == typ)].unique()

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
    df_sess = df_sess[df_sess[etime] < datetime.datetime(2020, 11, 9)]
    df_wifi = load('db_app_wifi_data_detail.csv', enSN)
    df_cell = load('db_app_modem_data_detail.csv', enSN)
    # df_audio = load('db_app_audio_detail.csv', enSN)
    # df_volume = load('db_audio_volume.csv', enSN) 
    # df_light = load('db_ambient_light_detail.csv', enSN)
    df_power_saving = load('db_power_saving.csv', enSN)
    df_battery = load('db_battery_detail.csv', enSN)
    df_disp = load('db_app_display_detail.csv', enSN)
    # df_tp = load('db_app_tp_detail.csv', enSN)
    # df_bt = load('db_app_bt_detail.csv', enSN)
    # df_camera = load('db_app_camera_detail.csv', enSN)
    # df_cpu = load('db_app_cpu_detail.csv', enSN)
    # df_gnss = load('db_app_gnss_detail.csv', enSN)
    # df_gpu = load('db_app_gpu_detail.csv', enSN)
    # df_sensor = load('db_app_sensor_detail.csv', enSN)
    series_dict = {}
    apps_work = load_app_class(enSN, '工作')
    apps_game = load_app_class(enSN, '游戏')
    apps_social = load_app_class(enSN, '社交')
    apps_video = load_app_class(enSN, '视频')
    apps_call = load_app_class(enSN, '通话')
    for i, row in tqdm(df_sess.iterrows(), total=df_sess.shape[0]):
        series_dict[i] = pd.DataFrame()
        curr = row[stime]
        
        D_duration = (row[stime] - row[ltime]).total_seconds()
        D_battery_used = get_between(df_battery, 'screen_on_gas_gauge', 'sum', row[ltime], row[stime])
        w_e, w_d = get_apps_between_with_duration(df_disp, 'energy', 'sum', row[ltime], row[stime], apps_work)
        g_e, g_d = get_apps_between_with_duration(df_disp, 'energy', 'sum', row[ltime], row[stime], apps_game)
        s_e, s_d = get_apps_between_with_duration(df_disp, 'energy', 'sum', row[ltime], row[stime], apps_social)
        v_e, v_d = get_apps_between_with_duration(df_disp, 'energy', 'sum', row[ltime], row[stime], apps_video)           
        c_e, c_d = get_apps_between_with_duration(df_disp, 'energy', 'sum', row[ltime], row[stime], apps_call)
        
        S_battery_used = 0
        S_wifi_upload = 0
        S_wifi_download = 0
        S_cell_upload = 0
        S_cell_download = 0
        
        count = 0
        while curr <= row[etime]:
            count += 1
            print(count)
            mode = get_mode(df_power_saving, 'mode', curr)
            item = {
                # GENERAL
                'hour': curr.hour,
                'minute': curr.minute,
                'second': curr.second,
                'day of week': curr.dayofweek,
                
                # DURING LAST SESSION
                'D duration': D_duration,
                'D battery used': D_battery_used,
                'D work duration': w_d,
                'D work energy': w_e,
                'D game duration': g_d,
                'D game energy': g_e,
                'D social duration': s_d,
                'D social energy': s_e,
                'D video duration': v_d,
                'D video energy': v_e,
                'D call duration': c_d,
                'D call energy': c_e,
                # 'D audio energy': get_between(df_audio, 'screen_on_energy', 'sum', row[ltime], row[stime]),
                # 'D display energy': get_between(df_disp, 'energy', 'sum', row[ltime], row[stime]),
                # 'D tp energy': get_between(df_tp, 'energy', 'sum', row[ltime], row[stime]),
                # 'D bt energy': get_between(df_bt, 'screen_on_energy', 'sum', row[ltime], row[stime]),
                # 'D camera energy': get_between(df_camera, 'screen_on_energy', 'sum', row[ltime], row[stime]),
                # 'D cpu energy': get_between(df_cpu, 'screen_on_energy', 'sum', row[ltime], row[stime]),
                # 'D gnss energy': get_between(df_gnss, 'screen_on_energy', 'sum', row[ltime], row[stime]),
                # 'D gpu energy': get_between(df_gpu, 'energy', 'sum', row[ltime], row[stime]),
                # 'D sensor energy': get_between(df_sensor, 'screen_on_energy', 'sum', row[ltime], row[stime]),
                
                # SINCE LAST SESSION 
                'S time': (curr - row[stime]).total_seconds(),
                'S battery used': S_battery_used,
                'S wifi data upload': S_wifi_upload,
                'S wifi data download': S_wifi_download,
                'S cellular data upload': S_cell_upload,
                'S cellular data download': S_cell_download,
                # 'S bt energy': get_between(df_bt, 'screen_off_energy', 'sum', row[stime], curr),
                # 'S audio energy': get_between(df_audio, 'screen_off_energy', 'sum', row[stime], curr),
                # 'S gnss energy': get_between(df_gnss, 'screen_off_energy', 'sum', row[stime], curr),
                # 'S gpu energy': get_between(df_gpu, 'energy', 'sum', row[stime], curr),
                # 'S sensor energy': get_between(df_sensor, 'screen_off_energy', 'sum', row[stime], curr),

                # NOW
                # 'C audio device':get_point(df_audio, 'device', curr),
                # 'C audio volume': get_point(df_volume, 'state', curr),
                # 'C ambient light': get_point(df_light, 'level', curr),
                'C mode 1': mode[0],
                'C mode 2': mode[1],
                'C mode 3': mode[2],
                'C mode 4': mode[3],
                'C charge': get_point(df_power_saving, 'charge', curr),
                'C # bright sessions': get_bright_session_num(df_sess, curr),
                'C # apps': get_app_num(df_disp, curr),
                
                # LABEL
                'time before bright': (row[etime] - curr).total_seconds()
                    }
            series_dict[i] = series_dict[i].append(item, ignore_index=True)
            curr += dt
            S_battery_used += get_between(df_battery, 'screen_off_gas_gauge', 'sum', curr - dt, curr)
            S_wifi_upload += get_between(df_wifi, 'screen_off_tx_bytes', 'sum', curr - dt, curr)
            S_wifi_download += get_between(df_wifi, 'screen_off_rx_bytes', 'sum', curr - dt, curr)
            S_cell_upload += get_between(df_cell, 'screen_off_tx_bytes', 'sum', curr - dt, curr)
            S_cell_download += get_between(df_cell, 'screen_off_rx_bytes', 'sum', curr - dt, curr)
    return series_dict

if __name__ == '__main__':
    for enSN in ['NOH000041']:
    #for enSN in ['ELS000040', 'ELS000043', 'ELS000063']:
        result = build_series(enSN)
        np.save('x_dict_10d_%s.npy' % enSN, result)