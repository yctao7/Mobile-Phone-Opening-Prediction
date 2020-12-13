import os
import pandas as pd
import matplotlib.pyplot as plt
from prep import normalize, prep_plot

comm_cols = ['productName', 'productVersion', 'timestamps',
             'formatted_start_time', 'formatted_end_time', 'name', 'enSN']
comm_cols2 = ['productName', 'productVersion', 'timestamps',
             'formatted_start_time', 'formatted_end_time', 'enSN']

targ_cols1 = [('foreground_duration', 'foreground_energy'),
              ('background_duration', 'background_energy'),
              ('screen_on_duration', 'screen_on_energy'),
              ('screen_off_duration', 'screen_off_energy')]

targ_cols2 = [(None, 'foreground_energy'), (None, 'background_energy'),
              (None, 'screen_on_energy'), (None, 'screen_off_energy')]

targ_cols3 = [(None, 'screen'),(None, 'low_tx_power_time'),
              (None, 'mid_tx_power_time'), (None, 'mid1_tx_power_time')
              ]
              ''' Only 4 subgraphs '''

targ_cols4 = [(None, 'mode'),(None, 'charge')]

targ_cols5 = [(None, 'screen_on_gas_gauge'), (None, 'screen_on_level'),
              (None, 'screen_off_gas_gauge'), (None, 'screen_off_level')]

targ_map = {'db_app_audio_detail.csv': targ_cols1, 'db_app_bt_detail.csv': targ_cols1,
            'db_app_camera_detail.csv': targ_cols1, 'db_app_cpu_detail.csv': targ_cols1,
            'db_app_ddr_detail.csv': targ_cols2, 'db_app_display_detail.csv': [('duration', 'energy')],
            'db_app_gnss_detail.csv': targ_cols1, 'db_app_gpu_detail.csv': [('usage', 'energy')],
            'db_app_modem_data_detail.csv': targ_cols2, 'db_app_sensor_detail.csv': targ_cols1,
            'db_app_tp_detail.csv': [(None, 'energy')], 'db_app_wifi_data_detail.csv': targ_cols2,
            'db_app_wifi_scan_detail.csv': targ_cols2, 'db_modem_lte_phy.csv': targ_cols3,
            'db_power_saving.csv':targ_cols4, 'db_battery_detail.csv':targ_cols5,
            'db_brightness_detail.csv':[(None,'brightness')]}

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
        df = pd.read_csv(os.path.join('./app', f))
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


def normal_app_without_name(enSN, files):
    dfn = pd.DataFrame()
    for f in files:
        df = pd.read_csv(os.path.join('./app', f))
        main_str = f.replace('detail.csv', '').replace('db_app_', '')
        if set(comm_cols2).issubset(set(df.columns)) and enSN in df['enSN'].unique():
            df = df[df['enSN'] == enSN]
            for duration, energy in targ_map[f]:
                new_cols = {energy: main_str + energy}
                if len(dfn) == 0:
                     dfn = normalize(df, energy, 'mean', duration).rename(columns=new_cols)
                else:
                     dfn = dfn.merge(normalize(df, energy, 'mean', duration).rename(columns=new_cols))
        else:
            for _, energy in targ_map[f]:
                dfn[main_str + energy] = 0.0
    return dfn

def darkness_interval_calculation(dataframe):
    for i in range(len(dfn) - 2, 0, -1):
        if dataframe.at[i, 'db_brightness_brightness'] == 0 and dataframe.at[i + 1, 'db_brightness_brightness'] != 0:
            dataframe.at[i, 'interval'] = 10
        elif dataframe.at[i, 'db_brightness_brightness'] == 0 and dataframe.at[i + 1, 'db_brightness_brightness'] == 0:
             dataframe.at[i, 'interval'] =  dataframe.at[i + 1, 'interval'] + 10
        else:
             dataframe.at[i, 'interval'] = 0




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
#     files = ['db_app_bt_detail.csv','db_app_modem_data_detail.csv',
#              'db_battery_detail.csv']
files = ['db_modem_lte_phy.csv','db_power_saving.csv',
        'db_battery_detail.csv','db_brightness_detail.csv']
name_files = ['db_app_bt_detail.csv']
dfn = normal_app_without_name(enSN, files)
dfn1 = normal_app(name, enSN, name_files)


# add time point
dfn['interval'] = 0

darkness_interval_calculation(dfn)

# Only use dark screen part.
dfn = dfn[dfn['db_brightness_brightness'] == 0]

plot(name, enSN, dfn, 'db_brightness_detail.csv')
plot(name, enSN, dfn1, 'db_app_bt_detail.csv')
#     plot(name, enSN, dfn, 'db_app_modem_data_detail.csv')
plot(name, enSN, dfn, 'db_modem_lte_phy.csv')
plot(name, enSN, dfn, 'db_power_saving.csv')
# plot(name, enSN, dfn, 'db_power_saving.csv')
# plot(name, enSN, dfn, 'db_app_wifi_scan_detail.csv')
# dfn
