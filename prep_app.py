import os
import pandas as pd
import matplotlib.pyplot as plt
from prep import normalize, prep_plot

comm_cols = ['productName', 'productVersion', 'timestamps',
             'formatted_start_time', 'formatted_end_time', 'name', 'enSN']

targ_cols1 = [('foreground_duration', 'foreground_energy'),
              ('background_duration', 'background_energy'),
              ('screen_on_duration', 'screen_on_energy'),
              ('screen_off_duration', 'screen_off_energy')]

targ_cols2 = [(None, 'foreground_energy'), (None, 'background_energy'),
              (None, 'screen_on_energy'), (None, 'screen_off_energy')]

targ_map = {'db_app_audio_detail.csv': targ_cols1, 'db_app_bt_detail.csv': targ_cols1,
            'db_app_camera_detail.csv': targ_cols1, 'db_app_cpu_detail.csv': targ_cols1,
            'db_app_ddr_detail.csv': targ_cols2, 'db_app_display_detail.csv': [('duration', 'energy')],
            'db_app_gnss_detail.csv': targ_cols1, 'db_app_gpu_detail.csv': [('usage', 'energy')],
            'db_app_modem_data_detail.csv': targ_cols2, 'db_app_sensor_detail.csv': targ_cols1,
            'db_app_tp_detail.csv': [(None, 'energy')], 'db_app_wifi_data_detail.csv': targ_cols2,
            'db_app_wifi_scan_detail.csv': targ_cols2}

def plot(name, enSN, dfn, f):
    pairs = []
    main_str = f.replace('detail.csv', '').replace('db_app_', '')
    for i, (_, energy) in enumerate(targ_map[f]):
        pairs.append((221+i, main_str + energy))
    plt.figure()
    for i, col in pairs:
        plt.subplot(i)
        plt.plot(*prep_plot(dfn, col))
        plt.title(name + '_' + enSN)
        plt.xlabel('Datetime')
        plt.ylabel(col)
    plt.show()

def normal_app(name, enSN, files):
    dfn = pd.DataFrame()
    for f in files:
        df = pd.read_csv(os.path.join('./data', f))
        main_str = f.replace('detail.csv', '').replace('db_app_', '')
        if set(comm_cols).issubset(set(df.columns)) and name in df['name'].unique() and enSN in df['enSN'].unique():
            df = df[(df['name'] == name) & (df['enSN'] == enSN)]
            for duration, energy in targ_map[f]:
                new_cols = {energy: main_str + energy}
                if len(dfn) == 0:
                    dfn = normalize(df, energy, 'sum', duration).rename(columns=new_cols)
                else:
                    dfn = dfn.merge(normalize(df, energy, 'sum', duration).rename(columns=new_cols))
        else:
            for _, energy in targ_map[f]:
                dfn[main_str + energy] = 0.0
    return dfn

if __name__ == '__main__':
    name = 'com.tencent.mm'
    enSN = 'ELS000040'
    # files = ['db_app_audio_detail.csv', 'db_app_bt_detail.csv',
    #           'db_app_camera_detail.csv', 'db_app_cpu_detail.csv',
    #           'db_app_ddr_detail.csv', 'db_app_display_detail.csv',
    #           'db_app_gnss_detail.csv', 'db_app_gpu_detail.csv',
    #           'db_app_modem_data_detail.csv', 'db_app_sensor_detail.csv',
    #           'db_app_tp_detail.csv', 'db_app_wifi_data_detail.csv',
    #           'db_app_wifi_scan_detail.csv']
    files = ['db_app_camera_detail.csv', 'db_app_bt_detail.csv',
             'db_app_gpu_detail.csv', 'db_app_wifi_scan_detail.csv']
    dfn = normal_app(name, enSN, files)
    plot(name, enSN, dfn, 'db_app_camera_detail.csv')
    plot(name, enSN, dfn, 'db_app_bt_detail.csv')
    plot(name, enSN, dfn, 'db_app_gpu_detail.csv')
    plot(name, enSN, dfn, 'db_app_wifi_scan_detail.csv')