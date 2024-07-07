import bisect
import os.path

import matplotlib.axis
import numpy as np
import pandas
import pandas as pd
import subprocess
import fire
import math
from io import StringIO

import matplotlib.pyplot as plt


pyrg_k1 = 8.986
pyrg_k2 = 1.028
sb = 5.6704e-8


def load_and_process_csv(csv_fn, temp_mappings=None, drop=None, baseline_env_Wm2=None, offset_secs=None):
    # load up the csv file
    df = pd.read_csv(csv_fn, encoding='utf8')

    # drop 1st 3 rows
    df = df[3:]

    # drop second "Date/Time" column
    df = df.drop(columns=['Date/Time.1'])

    # drop any in drop
    for todrop in (drop or []):
        df = df.drop(columns=[todrop])

    # convert "Date/Time" col data to timestamp
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    if offset_secs:
        # alter each date/time by this many seconds
        df['Date/Time'] = df['Date/Time'] + pd.to_timedelta(offset_secs, unit='s')

    rename = {
        'No.1': 'solar_in_V',
        'No.2': 'ir_net_V',
        'No.3': 'thermistor_V',
        'No.4': 'exc_V',
        **(temp_mappings or {}),
    }
    df = df.rename(columns=rename)

    for temp_key in temp_mappings.values():
        df[temp_key] = np.array(df[temp_key], dtype=np.float64)

    # remove rows with any nan values
    df = df.dropna()

    pyran_k1 = 24.51

    # add column which is this k1 * the solar in V
    solar_in_V = np.array(df['solar_in_V'], dtype=np.float64)
    df['solar_in_Wm2'] = solar_in_Wm2 = pyran_k1 * (solar_in_V*1000)
    df['solar_plus_env_Wm2'] = solar_in_Wm2 + (baseline_env_Wm2 or 0)

    # calculate thermistor T:
    excitation_voltage = np.array(df['exc_V'], dtype=np.float64)
    Vr = np.array(df['thermistor_V'], dtype=np.float64) / excitation_voltage
    Rt = 24900 * (Vr/(1 - Vr))
    A = 9.32794e-4
    B = 2.21451e-4
    C = 1.26233e-7
    thermistor_T_K = 1 / (A + B * np.log(Rt) + C * ((np.log(Rt))**3))
    df['thermistor_T_C'] = thermistor_T_K - 273.15

    ir_net_V = np.array(df['ir_net_V'], dtype=np.float64)
    df['ir_net_Wm2'] = ir_net_Wm2 = pyrg_k1 * (ir_net_V*1000)
    df['ir_in_Wm2'] = ir_in_Wm2 = ir_net_Wm2 + pyrg_k2 * sb * thermistor_T_K**4
    df['ir_out_Wm2'] = pyrg_k2 * sb * thermistor_T_K**4
    df['total_in_Wm2'] = solar_in_Wm2 + ir_in_Wm2
    df['solar_plus_net_ir_Wm2'] = solar_in_Wm2 + ir_net_Wm2

    df['max_implied_solar_C'] = (
        ((solar_in_Wm2) / sb)**(1/4) - 273.15
    )

    df['max_implied_total_C'] = (
        ((solar_in_Wm2 + ir_in_Wm2) / sb)**(1/4) - 273.15
    )

    df['max_implied_solar_env_C'] = (
        ((solar_in_Wm2+(baseline_env_Wm2 or 0)) / sb)**(1/4) - 273.15
    )

    # the conductive/convective loss must be equal to solar+ir
    df['cond_conv_loss_Wm2'] = solar_in_Wm2 + ir_net_Wm2

    return df


def load_and_process_logger_txt(txt_fn, temp_mappings=None, offset_secs=None):
    """Given filename to a file like this:

MN/AT  date      time     int    1ch     2ch    3ch    4ch    unit
AT  2024-05-08 12:33:33   1m            54.2    62.7    75.0   C
AT  2024-05-08 12:34:33   1m            50.7    63.4    75.7   C
AT  2024-05-08 12:35:33   1m            50.1    62.7    74.9   C
AT  2024-05-08 12:36:33   1m            51.8    62.1    75.6   C

    Return a dataframe with 'Date/Time', '1ch', '2ch', '3ch', and '4ch' columns,
    optionally mapped."""

    # load up the txt file:

    # read lines
    txt_lines = [l.strip() for l in open(txt_fn, encoding='utf8').readlines()]
    # get expected num cols from 2nd line
    num_cols = len(txt_lines[1].split())
    # drop any line besides first, that doesnt match
    txt_lines = [txt_lines[0]] + [line for line in txt_lines[1:] if len(line.split()) == num_cols]
    # put it together
    txt = '\n'.join(txt_lines)
    # read csv from this
    df = pd.read_csv(StringIO(txt), sep='\s+')

    # drop 1st row
    df = df[1:]

    # convert "date" and "time" to timestamp
    df['Date/Time'] = pd.to_datetime(df['date'] + ' ' + df['time'])

    if offset_secs:
        # alter each date/time by this many seconds
        df['Date/Time'] = df['Date/Time'] + pd.to_timedelta(offset_secs, unit='s')

    # drop "date" and "time" columns
    df = df.drop(columns=['date', 'time'])

    # drop columns with 'None' temp mapping
    temp_mappings = dict(temp_mappings or {})
    for col in list(df.columns):
        if col in temp_mappings and temp_mappings[col] is None:
            df = df.drop(columns=[col])
            del temp_mappings[col]

    df = df.rename(columns=temp_mappings)

    # for each row
    for temp_key in temp_mappings.values():
        # convert to float
        try:
            df[temp_key] = np.array(df[temp_key], dtype=np.float64)
        except Exception:
            # go through each row
            for i, row in df.iterrows():
                try:
                    float_val = float(row[temp_key])
                except Exception:
                    raise ValueError("Row #%d in file %s is bad has not float val %s" % (
                        i, txt_fn, row[temp_key],
                    ))

            raise ValueError("Converted all rows in file %s to float for column %s, dont know why failed" % (txt_fn, temp_key))

    return df


def load_and_process_logger_txts(txt_dir, temp_mappings=None, offset_secs=None):
    """Given a directory, load all *.TXT files in it and merge into one df."""

    # get all txt files
    txt_fns = [
        os.path.join(txt_dir, fn)
        for fn in os.listdir(txt_dir)
        if fn.endswith('.TXT')
    ]

    # load each txt file into a df
    dfs = [
        load_and_process_logger_txt(txt_fn, temp_mappings=temp_mappings, offset_secs=offset_secs)
        for txt_fn in txt_fns
    ]

    # merge all dfs into one
    df = pd.concat(dfs, ignore_index=True)

    # sort by Date/Time
    df = df.sort_values('Date/Time')

    return df


def calculate_emissivity(
    bottom_C,
    mid_C,
    ir_val,
    floor_sensor_dist_cm,
    vf_floor,
    n_segs=3,
):
    sb = 5.6704e-8

    # if bottom to mid_C is 2.5 cm and a smooth range
    # then calculate wall_at_sensor_C using floor_sensor_dist
    wall_at_sensor_C = bottom_C + (mid_C - bottom_C) * (floor_sensor_dist_cm / 2.5)

    raw_emission = vf_floor * sb * (bottom_C + 273.15)**4
    vf_wall = (1 - vf_floor)

    # split wall into 5 sub-segments
    for wall_i in range(n_segs):
        bottom_segment_C = bottom_C + (wall_at_sensor_C - bottom_C) * (wall_i / n_segs)
        top_segment_C = bottom_C + (wall_at_sensor_C - bottom_C) * ((wall_i+1) / n_segs)
        avg_segment_C = (bottom_segment_C + top_segment_C) / 2

        raw_emission += (vf_wall/n_segs) * sb * (avg_segment_C + 273.15)**4

    emissivity = ir_val / raw_emission
    return emissivity


def handle_implied_emissivity(
    dfs: list[pandas.DataFrame],
    bottom_keys: list[str],
    middle_keys: list[str],
    ir_key: str,
    vf_floor: float,
    ax,
):
    # make a new df taking Date/Time from first dfs
    target_df = dfs[0][['Date/Time']].copy()

    # make lists of datetimes of each of dfs
    dfs_datetimes = [
        list(df['Date/Time'])
        for df in dfs
    ]

    # dfs have different keys and diff date/times
    # for each date/time, pull nearest value from the df that has it
    # for each row in target_df
    n = 0
    n_rows = len(target_df)
    for tdf_i, row in target_df.iterrows():
        n += 1
        if n % 1000 == 0:
            print("Processing emissivity calc, row %d of %d" % (n, n_rows))

        nearest_vals_bot = []
        nearest_vals_mid = []
        ir_val = None
        for df, df_dts in zip(dfs, dfs_datetimes):
            # find nearest row in df, using binary search
            # on the Date/Time column
            i = bisect.bisect_left(df_dts, row['Date/Time'])
            if i == 0:
                nearest = df.iloc[0]
            elif i == len(df_dts):
                nearest = df.iloc[-1]
            else:
                # find which is closer
                if (row['Date/Time'] - df_dts[i-1]) < (df_dts[i] - row['Date/Time']):
                    nearest = df.iloc[i-1]
                else:
                    nearest = df.iloc[i]

            # for each key in bottom_keys, middle_keys
            for key in bottom_keys:
                if key not in df:
                    continue
                nearest_vals_bot.append(nearest[key])

            for key in middle_keys:
                if key not in df:
                    continue
                nearest_vals_mid.append(nearest[key])

            if ir_key in df:
                ir_val = nearest[ir_key]

        avg_bot = np.mean(nearest_vals_bot)
        avg_mid = np.mean(nearest_vals_mid)

        # add averages to target_df
        target_df.loc[tdf_i, 'avg_bot_C'] = avg_bot
        target_df.loc[tdf_i, 'avg_mid_C'] = avg_mid

        emissivity = calculate_emissivity(
            bottom_C=avg_bot,
            mid_C=avg_mid,
            ir_val=ir_val,
            floor_sensor_dist_cm=2.23,
            vf_floor=vf_floor,
        )
        target_df.loc[tdf_i, 'implied_emissivity'] = emissivity

    # now plot it on the axis

    # set axis stuff & plot
    # set x axis formatter to HH:MM
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

    # x axis label: time (HH:MM)
    ax.set_xlabel('Time (HH:MM)')

    # y axis label: W/m^2
    ax.set_ylabel('emissivity')

    # ax.set_ylim(0, 1)gri

    # rotate x axis labels
    plt.setp(ax.get_xticklabels(), rotation=45)

    ax.plot(target_df['Date/Time'], target_df['implied_emissivity'], label='Implied Emissivity')

    ax.legend(loc='lower right')
    ax.grid(axis='y', linestyle='-', linewidth=0.5)

    # draw hline at sensor emissivity of
    ax.axhline(pyrg_k2, color='brown', linestyle='--', linewidth=1)
    ax.text(
        pd.to_datetime('12 May 2024, 16:30'),
        pyrg_k2 + 0.02,
        "Pyrgeometer Calibration=%.3f" % pyrg_k2,
        rotation=0,
        color='brown',
    )


class CmdLine:
    def fix_encoding(self, csv_fn):
        # use 'uchardet' command
        result = subprocess.run(['uchardet', csv_fn], stdout=subprocess.PIPE)
        encoding = result.stdout.decode('utf8').strip()
        print("Detected encoding:", encoding)

        # open file with this encoding and output with utf8
        with open(csv_fn, encoding=encoding) as f:
            data = f.read()

        with open(csv_fn, 'w', encoding='utf8') as f:
            f.write(data)

        print("Fixed encoding for", csv_fn, "to utf8")

    def uncovered1(self, csv_fn='uncovered1.csv'):
        temp_mappings = {
            'No.5': 'top_air_C',
            'No.6': 'mid_air_C',
            'No.7': 'black_bottom_C',
        }

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)

        # export as xls
        df.to_excel('output.xlsx', index=False)

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#FF7F00')
        ax.get_lines()[-1].set_linewidth(1)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(1)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], linestyle=':', label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(1)

        # ax.plot(df['Date/Time'], df['ir_out_Wm2'], label='IR (out)')
        # ax.get_lines()[-1].set_color('indigo')
        #
        # ax.plot(df['Date/Time'], df['total_in_Wm2'], label='Total in')
        # ax.get_lines()[-1].set_color('darkgreen')

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-200, 1200)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.xticks(rotation=45)

        # on a separate Y axis, plot the thermistor_T_C
        ax2 = ax.twinx()
        ax2.plot(df['Date/Time'], df['thermistor_T_C'], color='coral', label='Thermistor')
        ax2.get_lines()[-1].set_linewidth(0.5)
        to_plot = df['max_implied_solar_C']
        # cut out any < 60C
        to_plot[to_plot < 60] = np.nan
        ax2.plot(df['Date/Time'], to_plot, color='black', linestyle='--', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(0.5)

        temp_colors = ['darkgoldenrod', 'gray', 'cyan', 'green']
        for temp_i, temp_key in enumerate(temp_mappings.values()):
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=temp_colors[temp_i])
            ax2.get_lines()[-1].set_linewidth(0.5)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='lower left')
        ax2.set_ylabel('Temperature (ºC)')
        # y axis: min 0, max 100
        ax2.set_ylim(0, 100)

        # add a thick blue vertical line at 2:00pm
        ax.axvline(pd.to_datetime('14:00'), color='blue', linestyle='--', linewidth=2)
        # and 3:45pm
        ax.axvline(pd.to_datetime('15:45'), color='blue', linestyle='--', linewidth=2)
        # and 5:04pm
        ax.axvline(pd.to_datetime('17:04'), color='blue', linestyle='--', linewidth=2)

        # title of graph: "Uncovered Box, 27 Apr 2024
        ax.set_title("Uncovered Box, 27 Apr 2024")

        plt.show()

    def seranwrap1(self, csv_fn):
        temp_mappings = {
            'No.5': 'top_air_C',
            'No.6': 'mid_air_C',
            'No.7': 'black_bottom_C',
        }

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)

        # export as xls
        df.to_excel('output.xlsx', index=False)

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(2, 1)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#FF7F00')
        ax.get_lines()[-1].set_linewidth(1)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(1)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(1)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(1)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(1)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-400, 1700)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the thermistor_T_C
        # ax2 = ax.twinx()
        ax2.plot(df['Date/Time'], df['thermistor_T_C'], color='coral', label='Thermistor')
        ax2.get_lines()[-1].set_linewidth(1)
        to_plot = df['max_implied_solar_C']
        # cut out any < 60C
        to_plot[to_plot < 60] = np.nan
        ax2.plot(df['Date/Time'], to_plot, color='#FF7F00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(1)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(1)

        temp_colors__wid = [
            ('cyan', 1),
            ('gray', 1),
            ('#7F0000', 3),
            ('green', 1),
        ]
        for temp_i, temp_key in enumerate(temp_mappings.values()):
            col, wid = temp_colors__wid[temp_i]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)


        # # add a thick blue vertical line at 2:00pm
        # ax.axvline(pd.to_datetime('14:00'), color='blue', linestyle='--', linewidth=2)
        # # and 3:45pm
        # ax.axvline(pd.to_datetime('15:45'), color='blue', linestyle='--', linewidth=2)
        # # and 5:04pm
        # ax.axvline(pd.to_datetime('17:04'), color='blue', linestyle='--', linewidth=2)

        # title of graph: "Uncovered Box, 27 Apr 2024
        ax.set_title("Seran-Wrap Box, 28 Apr 2024")

        plt.show()

    def seranwrap2(self, csv_fn):
        temp_mappings = {
            'No.5': 'top_C',
            'No.6': 'inside_middle_C',
            'No.7': 'black_bottom_C',
            'No.8': 'outside_air_C',
        }

        normal_line_width = 2
        emphasis_line_width = 3

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)

        # export as xls
        df.to_excel('output.xlsx', index=False)

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(2, 1)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-400, 1700)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the thermistor_T_C
        # ax2 = ax.twinx()
        ax2.plot(df['Date/Time'], df['thermistor_T_C'], color='coral', label='Thermistor')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)
        to_plot = df['max_implied_solar_C']
        # cut out any < 40C
        to_plot[to_plot < 40] = np.nan
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = [
            ('cyan', normal_line_width),
            ('gray', normal_line_width),
            ('#7F0000', emphasis_line_width),
            ('green', normal_line_width),
        ]
        for temp_i, temp_key in enumerate(temp_mappings.values()):
            col, wid = temp_colors__wid[temp_i]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color in [
            ('14:07', 'blue'),
            ('14:31', 'blue'),
            ('14:35', 'blue'),
            ('15:08', 'red'),
            ('15:10', 'blue'),
            ('16:31', 'red'),
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

        ax.set_title("Seran-Wrap & Glass Box, 29 Apr 2024")

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        plt.show()

    def threewrapscloudy(self, csv_fn):
        temp_mappings = {
            # 'No.5': 'top_C',
            'No.6': 'inside_middle_C',
            'No.7': 'black_bottom_C',
            'No.8': 'outside_air_C',
        }

        normal_line_width = 2
        emphasis_line_width = 3

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)

        # # drop anything before 13:00
        # df = df[df['Date/Time'] > pd.to_datetime('13:00')]

        # export as xls
        df.to_excel('output.xlsx', index=False)

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(2, 1)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        ax.get_lines()[-1].set_color('cyan')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-400, 1700)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the thermistor_T_C
        # ax2 = ax.twinx()
        ax2.plot(df['Date/Time'], df['thermistor_T_C'], color='coral', label='Thermistor')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)
        to_plot = df['max_implied_solar_C']
        # cut out any < 30C
        to_plot[to_plot < 30] = np.nan
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = [
            # ('cyan', normal_line_width),
            ('gray', normal_line_width),
            ('#7F0000', emphasis_line_width),
            ('green', normal_line_width),
        ]
        for temp_i, temp_key in enumerate(temp_mappings.values()):
            col, wid = temp_colors__wid[temp_i]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color in [
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

        ax.set_title("Three-layer Seran-Wrap Box, 02 May 2024")

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        plt.show()

    def onewrap(self, csv_fn):
        temp_mappings = {
            # 'No.5': 'top_C',
            'No.6': 'inside_middle_C',
            'No.7': 'black_bottom_C',
            'No.8': 'outside_air_C',
        }

        normal_line_width = 2
        emphasis_line_width = 3

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)

        # # drop anything before 13:00
        # df = df[df['Date/Time'] > pd.to_datetime('13:00')]

        # export as xls
        df.to_excel('output.xlsx', index=False)

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(2, 1)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        ax.get_lines()[-1].set_color('cyan')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-500, 1700)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the thermistor_T_C
        # ax2 = ax.twinx()
        ax2.plot(df['Date/Time'], df['thermistor_T_C'], color='coral', label='Thermistor')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)
        to_plot = df['max_implied_solar_C']
        # # cut out any < 30C
        # to_plot[to_plot < 30] = np.nan
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = [
            # ('cyan', normal_line_width),
            ('gray', normal_line_width),
            ('#7F0000', emphasis_line_width),
            ('green', normal_line_width),
        ]
        for temp_i, temp_key in enumerate(temp_mappings.values()):
            col, wid = temp_colors__wid[temp_i]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color in [
            ('12:40', 'blue'),
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

        ax.set_title("Three-then-one-layer Seran-Wrap Box, 03 May 2024")

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        plt.show()

    def wrapglass(self, csv_fn):
        temp_mappings = {
            'No.5': 'top_C',
            'No.6': 'inside_middle_C',
            'No.7': 'black_bottom_C',
            'No.8': 'outside_air_C',
        }

        normal_line_width = 2
        emphasis_line_width = 3

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)

        # # drop anything before 13:00
        # df = df[df['Date/Time'] > pd.to_datetime('13:00')]

        # export as xls
        df.to_excel('output.xlsx', index=False)

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(2, 1)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        ax.get_lines()[-1].set_color('cyan')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-500, 1700)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the thermistor_T_C
        # ax2 = ax.twinx()
        ax2.plot(df['Date/Time'], df['thermistor_T_C'], color='coral', label='Thermistor')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)
        to_plot = df['max_implied_solar_C']
        # # cut out any < 30C
        # to_plot[to_plot < 30] = np.nan
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = [
            ('cyan', normal_line_width),
            ('gray', normal_line_width),
            ('#7F0000', emphasis_line_width),
            ('green', normal_line_width),
        ]
        for temp_i, temp_key in enumerate(temp_mappings.values()):
            col, wid = temp_colors__wid[temp_i]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color in [
            # ('12:40', 'blue'),
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

        ax.set_title("Three-then-one-layer Seran-Wrap Box, 03 May 2024")

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        plt.show()

    def newboxopen(self, csv_fn):
        temp_mappings = {
            # 'No.5': 'top_C',
            # 'No.6': 'inside_middle_C',
            'No.7': 'outside_air_C',
            'No.8': 'black_bottom_C',
        }

        normal_line_width = 2
        emphasis_line_width = 3

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings, drop=['No.5', 'No.6'])

        # # drop anything before 13:00
        # df = df[df['Date/Time'] > pd.to_datetime('13:00')]

        # export as xls
        df.to_excel('output.xlsx', index=False)

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(2, 1)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        # ax.get_lines()[-1].set_color('darkgreen')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        # ax.get_lines()[-1].set_color('cyan')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-250, 1500)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the thermistor_T_C
        # ax2 = ax.twinx()
        ax2.plot(df['Date/Time'], df['thermistor_T_C'], color='coral', label='Thermistor')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)
        to_plot = df['max_implied_solar_C']
        # # cut out any < 30C
        # to_plot[to_plot < 30] = np.nan
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        # to_plot = df['max_implied_total_C']
        # ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        # ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = [
            # ('cyan', normal_line_width),
            # ('gray', normal_line_width),
            ('green', normal_line_width),
            ('#7F0000', emphasis_line_width),
        ]
        for temp_i, temp_key in enumerate(temp_mappings.values()):
            col, wid = temp_colors__wid[temp_i]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color in [
            ('13:12', 'blue'),  # tilted up
            ('14:13', 'blue'),  # added edge
            ('15:18', 'blue'),  # moved
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

        ax.set_title("05 May 2024; Box v2, 1 edge, uncovered")

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        plt.show()

    def newboxseran(self, csv_fn='~/Downloads/pyrgepyran/newboxseran.csv'):
        csv_fn = os.path.expanduser(csv_fn)

        temp_mappings = {
            'No.5': 'inside_air_C',
            'No.6': 'aluminum_C',
            'No.7': 'styro_5mm_C',
            'No.8': 'styro_surf_C',
        }

        normal_line_width = 2
        emphasis_line_width = 3

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)

        # # drop anything before 13:00
        # df = df[df['Date/Time'] > pd.to_datetime('13:00')]

        # export as xls
        df.to_excel('output.xlsx', index=False)

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(2, 1)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        # ax.get_lines()[-1].set_color('cyan')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-500, 1750)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the thermistor_T_C
        # ax2 = ax.twinx()
        ax2.plot(df['Date/Time'], df['thermistor_T_C'], color='coral', label='Thermistor')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)
        to_plot = df['max_implied_solar_C']
        # # cut out any < 30C
        # to_plot[to_plot < 30] = np.nan
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = {
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
        }
        for temp_key in temp_mappings.values():
            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color in [
            ('14:55', 'blue'),  # final movement
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

        ax.set_title("06 May 2024; Box v2, 1 seran")

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        plt.show()

    def newbox2seran(self, csv_fn='~/Downloads/pyrgepyran/newbox2seran.csv'):
        csv_fn = os.path.expanduser(csv_fn)

        temp_mappings = {
            'No.5': 'inside_air_C',
            'No.6': '3rd_layer_C',
            'No.7': '2nd_layer_C',
            'No.8': 'top_C',
        }

        normal_line_width = 2
        emphasis_line_width = 3

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)

        # # drop anything before 13:00
        # df = df[df['Date/Time'] > pd.to_datetime('13:00')]

        # export as xls
        df.to_excel('output.xlsx', index=False)

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        import matplotlib.pyplot as plt
        fig, (ax, ax2) = plt.subplots(2, 1)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        # ax.get_lines()[-1].set_color('cyan')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-500, 1750)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the temps
        to_plot = df['max_implied_solar_C']
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = {
            'thermistor_T_C': ('coral', normal_line_width),
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
            '3rd_layer_C': ('gray', normal_line_width),
            # 'No.7': '2nd_layer_C',
            '2nd_layer_C': ('brown', normal_line_width),
            'top_C': ('#4F0000', emphasis_line_width),
        }
        for temp_key in temp_colors__wid:
            # if not in df, skip
            if temp_key not in df:
                continue

            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color, reason in [
            ('11:42', 'blue', 'open'),
            ('12:22', 'blue', '1st seran'),
            ('13:13', 'blue', '2nd seran'),
            ('14:09', 'orange', 'robe belt'),
            ('14:55', 'orange', 'final adjust'),
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

            # draw label of the reason on top graph
            ax.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=60),
                1600,
                reason,
                rotation=0,
                color=color,
            )

        ax.set_title("07 May 2024; Box v3, open, 1, and 2 seran")

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        plt.show()

    def box3_2seran_allday(self, csv_fn='~/Downloads/pyrgepyran/box3_2seran_allday/box3_2seran_allday.csv'):
        import matplotlib.pyplot as plt

        title = "08 May 2024; Box v3, 2 seran, all day unmoving"

        csv_fn = os.path.expanduser(csv_fn)
        run_dir = os.path.dirname(csv_fn)
        run_bit = os.path.basename(csv_fn).rsplit('.', 1)[0]

        normal_line_width = 2
        emphasis_line_width = 3

        temp_mappings = {
            'No.5': 'inside_air_C',
            'No.6': '3rd_layer_C',
            'No.7': '2nd_layer_C',
            'No.8': 'top_layer_C',
        }

        logger_temp_mappings = {
            '1ch': 'seran2_layer_C',
            '2ch': 'seran1_air_C',
            '3ch': 'seran1_layer_C',
            '4ch': None,
        }

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)
        df_logger = load_and_process_logger_txts(run_dir, temp_mappings=logger_temp_mappings)

        # # drop anything before 13:00
        # df = df[df['Date/Time'] > pd.to_datetime('13:00')]

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax.set_title(title)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        # ax.get_lines()[-1].set_color('cyan')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-500, 1750)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the temps
        to_plot = df['max_implied_solar_C']
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = {
            'thermistor_T_C': ('coral', normal_line_width),
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
            '3rd_layer_C': ('gray', normal_line_width),
            # 'No.7': '2nd_layer_C',
            '2nd_layer_C': ('brown', normal_line_width),
            'top_layer_C': ('#4F0000', emphasis_line_width),
        }
        for temp_key in temp_colors__wid:
            # if not in df, skip
            if temp_key not in df:
                continue

            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        logger_temp_colors__wid = {
            'seran2_layer_C': ('#0000bb', normal_line_width),
            'seran1_air_C': ('#000088', normal_line_width),
            'seran1_layer_C': ('#000033', normal_line_width),
        }
        for temp_key in logger_temp_colors__wid:
            # if not in df, skip
            if temp_key not in df_logger:
                continue

            col, wid = logger_temp_colors__wid[temp_key]
            ax2.plot(df_logger['Date/Time'], df_logger[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color, reason in [
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

            # draw label of the reason on top graph
            ax.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=60),
                1600,
                reason,
                rotation=0,
                color=color,
            )

        # get max value of top_layer_C
        max_temp = df['top_layer_C'].max()
        # draw dashed horizontal top_layer_C-colored line there
        ax2.axhline(max_temp, color=temp_colors__wid['top_layer_C'][0], linestyle='--', linewidth=2)

        # draw vertical line where the top layer C is this max val
        ax2.axvline(
            df[df['top_layer_C'] == max_temp]['Date/Time'].iloc[0],
            color=temp_colors__wid['top_layer_C'][0],
            linestyle='--',
            linewidth=2,
        )

        # write text of what time that is
        # draw text above the line, "Max Temp=..."
        ax2.text(
            df['Date/Time'].iloc[0],
            max_temp + 5,
            f"Max top_layer_C={max_temp:.2f}ºC @ {df[df['top_layer_C'] == max_temp]['Date/Time'].iloc[0]}",
            rotation=0,
            color=temp_colors__wid['top_layer_C'][0],
        )


        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        # save to .png
        plt.tight_layout()
        dest_fn = os.path.join(run_dir, run_bit + '.png')
        plt.savefig(
            dest_fn,
            dpi=300,
        )

        plt.show()

    def box3_2seran_allday2(self, csv_fn='~/Downloads/pyrgepyran/box3_2seran_allday2/box3_2seran_allday2.csv'):
        import matplotlib.pyplot as plt

        title = "09 May 2024; Box v3, 2 seran, all day unmoving 2"

        csv_fn = os.path.expanduser(csv_fn)
        run_dir = os.path.dirname(csv_fn)
        run_bit = os.path.basename(csv_fn).rsplit('.', 1)[0]

        normal_line_width = 2
        emphasis_line_width = 3

        temp_mappings = {
            'No.5': 'inside_air_C',
            'No.6': '3rd_layer_C',
            'No.7': '2nd_layer_C',
            'No.8': 'top_layer_C',
        }

        logger_temp_mappings = {
            '1ch': 'seran2_layer_C',
            '2ch': 'seran1_air_C',
            '3ch': 'seran1_layer_C',
            '4ch': None,
        }

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)
        df_logger = load_and_process_logger_txts(run_dir, temp_mappings=logger_temp_mappings)

        # # drop anything before 13:00
        # df = df[df['Date/Time'] > pd.to_datetime('13:00')]

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax.set_title(title)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        # ax.get_lines()[-1].set_color('cyan')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-500, 1750)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the temps
        to_plot = df['max_implied_solar_C']
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = {
            'thermistor_T_C': ('coral', normal_line_width),
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
            '3rd_layer_C': ('gray', normal_line_width),
            # 'No.7': '2nd_layer_C',
            '2nd_layer_C': ('brown', normal_line_width),
            'top_layer_C': ('#4F0000', emphasis_line_width),
        }
        for temp_key in temp_colors__wid:
            # if not in df, skip
            if temp_key not in df:
                continue

            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        logger_temp_colors__wid = {
            'seran2_layer_C': ('#0000bb', normal_line_width),
            'seran1_air_C': ('#000088', normal_line_width),
            'seran1_layer_C': ('#000033', normal_line_width),
        }
        for temp_key in logger_temp_colors__wid:
            # if not in df, skip
            if temp_key not in df_logger:
                continue

            col, wid = logger_temp_colors__wid[temp_key]
            ax2.plot(df_logger['Date/Time'], df_logger[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color, reason in [
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

            # draw label of the reason on top graph
            ax.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=60),
                1600,
                reason,
                rotation=0,
                color=color,
            )

        # get max value of top_layer_C
        max_temp = df['top_layer_C'].max()
        # draw dashed horizontal top_layer_C-colored line there
        ax2.axhline(max_temp, color=temp_colors__wid['top_layer_C'][0], linestyle='--', linewidth=2)

        # draw vertical line where the top layer C is this max val
        ax2.axvline(
            df[df['top_layer_C'] == max_temp]['Date/Time'].iloc[0],
            color=temp_colors__wid['top_layer_C'][0],
            linestyle='--',
            linewidth=2,
        )

        # write text of what time that is
        # draw text above the line, "Max Temp=..."
        ax2.text(
            df['Date/Time'].iloc[0],
            max_temp + 5,
            f"Max top_layer_C={max_temp:.2f}ºC @ {df[df['top_layer_C'] == max_temp]['Date/Time'].iloc[0]}",
            rotation=0,
            color=temp_colors__wid['top_layer_C'][0],
        )

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        # save to .png
        plt.tight_layout()
        dest_fn = os.path.join(run_dir, run_bit + '.png')
        plt.savefig(
            dest_fn,
            dpi=300,
        )

        plt.show()

    def box3_open_allday(self, csv_fn='~/Downloads/pyrgepyran/box3_open_allday/box3_open_allday.csv'):
        import matplotlib.pyplot as plt

        title = "10 May 2024; Box v3, opan, all day unmoving"

        csv_fn = os.path.expanduser(csv_fn)
        run_dir = os.path.dirname(csv_fn)
        run_bit = os.path.basename(csv_fn).rsplit('.', 1)[0]

        normal_line_width = 2
        emphasis_line_width = 3

        temp_mappings = {
            'No.5': 'inside_air_C',
            'No.6': '3rd_layer_C',
            'No.7': '2nd_layer_C',
            'No.8': 'top_layer_C',
        }

        logger_temp_mappings = {
        }

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)
        df_logger = None

        # # drop anything before 13:00
        # df = df[df['Date/Time'] > pd.to_datetime('13:00')]

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax.set_title(title)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        # ax.get_lines()[-1].set_color('cyan')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-500, 1750)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the temps
        to_plot = df['max_implied_solar_C']
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = {
            'thermistor_T_C': ('coral', normal_line_width),
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
            '3rd_layer_C': ('gray', normal_line_width),
            # 'No.7': '2nd_layer_C',
            '2nd_layer_C': ('brown', normal_line_width),
            'top_layer_C': ('#4F0000', emphasis_line_width),
        }
        for temp_key in temp_colors__wid:
            # if not in df, skip
            if temp_key not in df:
                continue

            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        if df_logger:
            logger_temp_colors__wid = {
                'seran2_layer_C': ('#0000bb', normal_line_width),
                'seran1_air_C': ('#000088', normal_line_width),
                'seran1_layer_C': ('#000033', normal_line_width),
            }
            for temp_key in logger_temp_colors__wid:
                # if not in df, skip
                if temp_key not in df_logger:
                    continue

                col, wid = logger_temp_colors__wid[temp_key]
                ax2.plot(df_logger['Date/Time'], df_logger[temp_key], label=temp_key, color=col)
                ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color, reason in [
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

            # draw label of the reason on top graph
            ax.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=60),
                1600,
                reason,
                rotation=0,
                color=color,
            )

        # get max value of top_layer_C
        max_temp = df['top_layer_C'].max()
        # draw dashed horizontal top_layer_C-colored line there
        ax2.axhline(max_temp, color=temp_colors__wid['top_layer_C'][0], linestyle='--', linewidth=2)

        # draw vertical line where the top layer C is this max val
        ax2.axvline(
            df[df['top_layer_C'] == max_temp]['Date/Time'].iloc[0],
            color=temp_colors__wid['top_layer_C'][0],
            linestyle='--',
            linewidth=2,
        )

        # write text of what time that is
        # draw text above the line, "Max Temp=..."
        ax2.text(
            df['Date/Time'].iloc[0],
            max_temp + 5,
            f"Max top_layer_C={max_temp:.2f}ºC @ {df[df['top_layer_C'] == max_temp]['Date/Time'].iloc[0]}",
            rotation=0,
            color=temp_colors__wid['top_layer_C'][0],
        )

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        # save to .png
        plt.tight_layout()
        dest_fn = os.path.join(run_dir, run_bit + '.png')
        plt.savefig(
            dest_fn,
            dpi=300,
        )

        plt.show()

    def box3_3seran_allday(self, csv_fn='~/Downloads/pyrgepyran/box3_3seran_allday/box3_3seran_allday.csv'):
        import matplotlib.pyplot as plt

        title = "11 May 2024; Box v3, 3 serans, all day unmoving"

        csv_fn = os.path.expanduser(csv_fn)
        run_dir = os.path.dirname(csv_fn)
        run_bit = os.path.basename(csv_fn).rsplit('.', 1)[0]

        normal_line_width = 2
        emphasis_line_width = 3

        temp_mappings = {
            'No.5': 'inside_air_C',
            'No.6': '3rd_layer_C',
            'No.7': '2nd_layer_C',
            'No.8': 'top_layer_C',
        }

        logger_temp_mappings = {
            '1ch': 'seran2_air_C',
            '2ch': 'seran2_layer_C',
            '3ch': 'seran3_air_C',
            '4ch': 'seran3_layer_C',
        }

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings)
        df_logger = load_and_process_logger_txts(run_dir, temp_mappings=logger_temp_mappings)

        # # drop anything before 13:00
        # df = df[df['Date/Time'] > pd.to_datetime('13:00')]

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 10))

        ax.set_title(title)

        to_plot = df['solar_in_Wm2']
        # cut off values < 700
        # to_plot[to_plot < 700] = np.nan
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Downwelling IR')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Upwelling IR')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        ax.get_lines()[-1].set_color('darkgreen')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        # ax.get_lines()[-1].set_color('cyan')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-500, 1750)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the temps
        to_plot = df['max_implied_solar_C']
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_total_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = {
            'thermistor_T_C': ('coral', normal_line_width),
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
            '3rd_layer_C': ('gray', normal_line_width),
            # 'No.7': '2nd_layer_C',
            '2nd_layer_C': ('brown', normal_line_width),
            'top_layer_C': ('#4F0000', emphasis_line_width),
        }
        for temp_key in temp_colors__wid:
            # if not in df, skip
            if temp_key not in df:
                continue

            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        if df_logger is not None:
            logger_temp_colors__wid = {
                'seran2_air_C': ('#0000ff', normal_line_width),
                'seran2_layer_C': ('#0000bb', normal_line_width),
                'seran3_air_C': ('#000088', normal_line_width),
                'seran3_layer_C': ('#000033', normal_line_width),
            }
            for temp_key in logger_temp_colors__wid:
                # if not in df, skip
                if temp_key not in df_logger:
                    continue

                col, wid = logger_temp_colors__wid[temp_key]
                ax2.plot(df_logger['Date/Time'], df_logger[temp_key], label=temp_key, color=col)
                ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 150)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color, reason in [
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=2)

            # draw label of the reason on top graph
            ax.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=60),
                1600,
                reason,
                rotation=0,
                color=color,
            )

        # get max value of top_layer_C
        max_temp = df['top_layer_C'].max()
        # draw dashed horizontal top_layer_C-colored line there
        ax2.axhline(max_temp, color=temp_colors__wid['top_layer_C'][0], linestyle='--', linewidth=2)

        # draw vertical line where the top layer C is this max val
        ax2.axvline(
            df[df['top_layer_C'] == max_temp]['Date/Time'].iloc[0],
            color=temp_colors__wid['top_layer_C'][0],
            linestyle='--',
            linewidth=2,
        )

        # write text of what time that is
        # draw text above the line, "Max Temp=..."
        ax2.text(
            df['Date/Time'].iloc[0],
            max_temp + 5,
            f"Max top_layer_C={max_temp:.2f}ºC @ {df[df['top_layer_C'] == max_temp]['Date/Time'].iloc[0]}",
            rotation=0,
            color=temp_colors__wid['top_layer_C'][0],
        )

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        # save to .png
        plt.tight_layout()
        dest_fn = os.path.join(run_dir, run_bit + '.png')
        plt.savefig(
            dest_fn,
            dpi=300,
        )

        plt.show()

    def box3_3seran_revir(self, csv_fn='~/Downloads/pyrgepyran/box3_3seran_revir/box3_3seran_revir.csv'):
        import matplotlib.pyplot as plt

        title = "12 May 2024; Box v3, 3 serans, pyrgeometer downwards"

        csv_fn = os.path.expanduser(csv_fn)
        run_dir = os.path.dirname(csv_fn)
        run_bit = os.path.basename(csv_fn).rsplit('.', 1)[0]

        normal_line_width = 2
        emphasis_line_width = 3

        baseline_env_Wm2 = 500

        temp_mappings = {
            'No.5': 'inside_air_C',
            'No.6': '3rd_layer_C',
            'No.7': '2nd_layer_C',
            'No.8': 'top_layer_uppermid_C',
        }

        logger_temp_mappings = {
            '1ch': 'seran3_air_C',
            '2ch': 'seran3_layer_C',
            '3ch': 'top_layer_lowerleft_C',
            '4ch': 'top_layer_lowermid_C',
        }

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings, baseline_env_Wm2=baseline_env_Wm2)
        df_logger = load_and_process_logger_txts(run_dir, temp_mappings=logger_temp_mappings)

        # drop anything before 11:03
        df = df[df['Date/Time'] > pd.to_datetime('12 May 2024, 11:03')]
        df_logger = df_logger[df_logger['Date/Time'] > pd.to_datetime('12 May 2024, 11:03')]

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        fig, (ax, ax2, ax3) = plt.subplots(
            3,
            1,
            figsize=(14, 10),
            height_ratios=[2, 2, 1],
        )

        ax.set_title(title)

        to_plot = df['solar_in_Wm2']
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Upwards IR (from bottom)')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Downwards IR (from pyrg)')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['solar_plus_env_Wm2'], linestyle=':', label='Solar + Env')
        ax.get_lines()[-1].set_color('green')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        # ax.get_lines()[-1].set_color('darkgreen')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        # ax.get_lines()[-1].set_color('cyan')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-500, 1500)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        ax.axhline(baseline_env_Wm2, color='brown', linestyle='--', linewidth=1)
        # text above baseline env saying baseline env
        ax.text(
            pd.to_datetime('12 May 2024, 16:30'),
            baseline_env_Wm2 + 20,
            "Baseline Env=%.0f W/m^2" % baseline_env_Wm2,
            rotation=0,
            color='brown',
        )

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the temps
        to_plot = df['max_implied_solar_C']
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        to_plot = df['max_implied_solar_env_C']
        ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Solar+Env')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        # to_plot = df['max_implied_total_C']
        # ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        # ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = {
            'thermistor_T_C': ('#ff33ff', normal_line_width),
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
            # '3rd_layer_C': ('gray', normal_line_width),
            # 'No.7': '2nd_layer_C',
            # '2nd_layer_C': ('brown', normal_line_width),
            'top_layer_C': ('#4F0000', emphasis_line_width),
            'top_layer_uppermid_C': ('#4F0000', normal_line_width),
            'seran2_air_C': ('#7777ff', normal_line_width),
            'seran2_layer_C': ('#3333ff', normal_line_width),
            # 'seran3_air_C': ('#2222ff', normal_line_width),
            'seran3_layer_C': ('#0000bf', normal_line_width),
            'top_layer_lowermid_C': ('#FF0000', normal_line_width),
            'top_layer_lowerleft_C': ('#FF7777', normal_line_width),
        }
        for temp_key in temp_colors__wid:
            # if not in df, skip
            if temp_key not in df:
                continue

            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        if df_logger is not None:
            for temp_key in temp_colors__wid:
                # if not in df, skip
                if temp_key not in df_logger:
                    continue

                col, wid = temp_colors__wid[temp_key]
                ax2.plot(df_logger['Date/Time'], df_logger[temp_key], label=temp_key, color=col)
                ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(0, 140)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color, reason, linewidth in [
            ('12 May 2024, 11:10', 'blue', 'start', 2),
            ('12 May 2024, 11:23', 'orange', '', 1),
            ('12 May 2024, 11:36', 'orange', '', 1),
            ('12 May 2024, 11:46', 'orange', '', 1),
            ('12 May 2024, 11:54', 'orange', '', 1),
            ('12 May 2024, 12:07', 'orange', '', 1),
            ('12 May 2024, 12:22', 'orange', '', 1),
            ('12 May 2024, 12:38', 'orange', '', 1),
            ('12 May 2024, 12:55', 'orange', '', 1),
            ('12 May 2024, 13:06', 'orange', '', 1),
            ('12 May 2024, 13:46', 'orange', '', 1),
            ('12 May 2024, 13:59', 'blue', 'cover', 2),
            ('12 May 2024, 14:18', 'orange', 'turn to face sun', 1),
            ('12 May 2024, 15:17', 'orange', '', 1),
            ('12 May 2024, 15:54', 'orange', '', 1),
            ('12 May 2024, 16:29', 'blue', 'lay flat', 2),
        ]:
            for a in [ax, ax2, ax3]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=linewidth)

            # draw label of the reason on top graph
            ax.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=60),
                1430,
                reason,
                rotation=0,
                color=color,
            )
            # and bottom
            ax2.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=60),
                135,
                reason,
                rotation=0,
                color=color,
            )

        # get max value of top_layer_C
        max_temp, maxdf, maxkey = max(
            (df['top_layer_uppermid_C'].max(), df, 'top_layer_uppermid_C'),
            (df_logger['top_layer_lowerleft_C'].max(), df_logger, 'top_layer_lowerleft_C'),
            (df_logger['top_layer_lowermid_C'].max(), df_logger, 'top_layer_lowermid_C'),
        )
        # draw dashed horizontal top_layer_C-colored line there
        ax2.axhline(max_temp, color=temp_colors__wid[maxkey][0], linestyle='--', linewidth=2)

        # draw vertical line where the top layer C is this max val
        ax2.axvline(
            maxdf[maxdf[maxkey] == max_temp]['Date/Time'].iloc[0],
            color=temp_colors__wid[maxkey][0],
            linestyle='--',
            linewidth=2,
        )

        # write text of what time that is
        # draw text above the line, "Max Temp=..."
        ax2.text(
            maxdf['Date/Time'].iloc[0] + pd.to_timedelta('20m'),
            5,
            f"Max {maxkey}  ={max_temp:.1f}ºC",
            rotation=0,
            color=temp_colors__wid[maxkey][0],
        )

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        # --------------- implied emissivity -------------------------
        handle_implied_emissivity(
            dfs=[df, df_logger],
            bottom_keys=['top_layer_uppermid_C', 'top_layer_lowerleft_C', 'top_layer_lowermid_C'],
            middle_keys=['inside_air_C'],
            ir_key='ir_in_Wm2',
            vf_floor=0.46,
            ax=ax3,
        )

        # save to .png
        plt.tight_layout()
        dest_fn = os.path.join(run_dir, run_bit + '.png')
        plt.savefig(
            dest_fn,
            dpi=300,
        )

        plt.show()

    def box3_3seran_cover_allday(self, csv_fn='~/Downloads/pyrgepyran/box3_3seran_cover_allday/box3_3seran_cover_allday.csv'):
        import matplotlib.pyplot as plt

        title = "12 May 2024; Box v3, 3 serans, pyrgeometer downwards, enclosed"

        csv_fn = os.path.expanduser(csv_fn)
        run_dir = os.path.dirname(csv_fn)
        run_bit = os.path.basename(csv_fn).rsplit('.', 1)[0]

        normal_line_width = 2
        emphasis_line_width = 3

        baseline_env_Wm2 = 540

        temp_mappings = {
            'No.5': 'inside_air_C',
            'No.6': '3rd_layer_C',
            'No.7': '2nd_layer_C',
            'No.8': 'top_layer_uppermid_C',
        }

        logger_temp_mappings = {
            '1ch': 'seran3_air_C',
            '2ch': 'seran3_layer_C',
            '3ch': 'top_layer_lowerleft_C',
            '4ch': 'top_layer_lowermid_C',
        }

        df = load_and_process_csv(csv_fn, temp_mappings=temp_mappings, baseline_env_Wm2=baseline_env_Wm2)
        df_logger = load_and_process_logger_txts(run_dir, temp_mappings=logger_temp_mappings)

        # drop anything before 02:00
        df = df[df['Date/Time'] > pd.to_datetime('13 May 2024, 02:00')]
        df_logger = df_logger[df_logger['Date/Time'] > pd.to_datetime('13 May 2024, 02:00')]

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        fig, (ax, ax2) = plt.subplots(
            2,
            1,
            figsize=(12, 10),
        )

        ax.set_title(title)

        to_plot = df['solar_in_Wm2']
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax.get_lines()[-1].set_color('red')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_in_Wm2'], linestyle=':', label='Upwards IR (from bottom)')
        ax.get_lines()[-1].set_color('darkred')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.plot(df['Date/Time'], df['ir_out_Wm2'], linestyle=':', label='Downwards IR (from pyrg)')
        ax.get_lines()[-1].set_color('indigo')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['solar_plus_env_Wm2'], linestyle=':', label='Solar + Env')
        # ax.get_lines()[-1].set_color('green')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['total_in_Wm2'], linestyle=':', label='Solar + Downwelling IR')
        # ax.get_lines()[-1].set_color('darkgreen')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.plot(df['Date/Time'], df['cond_conv_loss_Wm2'], linestyle=':', label='Net Rad (Solar+Net IR)')
        # ax.get_lines()[-1].set_color('cyan')
        # ax.get_lines()[-1].set_linewidth(normal_line_width)

        ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        ax.set_ylim(-100, 800)

        # draw red horizontal line at 0
        ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # ax.axhline(baseline_env_Wm2, color='brown', linestyle='--', linewidth=1)
        # # text above baseline env saying baseline env
        # ax.text(
        #     pd.to_datetime('12 May 2024, 16:30'),
        #     baseline_env_Wm2 + 20,
        #     "Baseline Env=%.0f W/m^2" % baseline_env_Wm2,
        #     rotation=0,
        #     color='brown',
        # )

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # on a separate Y axis, plot the temps
        to_plot = df['max_implied_solar_C']
        ax2.plot(df['Date/Time'], to_plot, color='#DFDF00', linestyle=':', label='Max Implied Solar')
        ax2.get_lines()[-1].set_linewidth(normal_line_width)

        # to_plot = df['max_implied_solar_env_C']
        # ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Solar+Env')
        # ax2.get_lines()[-1].set_linewidth(normal_line_width)

        # to_plot = df['max_implied_total_C']
        # ax2.plot(df['Date/Time'], to_plot, color='darkgreen', linestyle=':', label='Max Implied Total')
        # ax2.get_lines()[-1].set_linewidth(normal_line_width)

        temp_colors__wid = {
            'thermistor_T_C': ('#ff33ff', normal_line_width),
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
            # '3rd_layer_C': ('gray', normal_line_width),
            # 'No.7': '2nd_layer_C',
            # '2nd_layer_C': ('brown', normal_line_width),
            'top_layer_C': ('#4F0000', emphasis_line_width),
            'top_layer_uppermid_C': ('#4F0000', normal_line_width),
            'seran2_air_C': ('#7777ff', normal_line_width),
            'seran2_layer_C': ('#3333ff', normal_line_width),
            'seran3_air_C': ('#2222ff', normal_line_width),
            # 'seran3_layer_C': ('#0000bf', normal_line_width),
            'top_layer_lowermid_C': ('#FF0000', normal_line_width),
            'top_layer_lowerleft_C': ('#FF7777', normal_line_width),
        }
        for temp_key in temp_colors__wid:
            # if not in df, skip
            if temp_key not in df:
                continue

            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        if df_logger is not None:
            for temp_key in temp_colors__wid:
                # if not in df, skip
                if temp_key not in df_logger:
                    continue

                col, wid = temp_colors__wid[temp_key]
                ax2.plot(df_logger['Date/Time'], df_logger[temp_key], label=temp_key, color=col)
                ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))
        ax2.set_ylabel('Temperature (ºC)')
        ax2.set_ylim(10, 60)
        plt.setp(ax2.get_xticklabels(), rotation=45)

        for adj_time, color, reason, linewidth in [
            # ('13 May 2024, 17:57', 'blue', 'tip', 1),
            # ('13 May 2024, 19:03', 'blue', 'fell', 1),
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=linewidth)

            # draw label of the reason on top graph
            ax.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=60),
                750,
                reason,
                rotation=0,
                color=color,
            )
            # and bottom
            ax2.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=60),
                50,
                reason,
                rotation=0,
                color=color,
            )

        # get max value of top_layer_C
        max_temp, maxdf, maxkey = max(
            (df['top_layer_uppermid_C'].max(), df, 'top_layer_uppermid_C'),
            (df_logger['top_layer_lowerleft_C'].max(), df_logger, 'top_layer_lowerleft_C'),
            (df_logger['top_layer_lowermid_C'].max(), df_logger, 'top_layer_lowermid_C'),
        )
        # draw dashed horizontal top_layer_C-colored line there
        ax2.axhline(max_temp, color=temp_colors__wid[maxkey][0], linestyle='--', linewidth=2)

        # draw vertical line where the top layer C is this max val
        ax2.axvline(
            maxdf[maxdf[maxkey] == max_temp]['Date/Time'].iloc[0],
            color=temp_colors__wid[maxkey][0],
            linestyle='--',
            linewidth=2,
        )

        # write text of what time that is
        # draw text above the line, "Max Temp=..."
        ax2.text(
            maxdf['Date/Time'].iloc[0] + pd.to_timedelta('20m'),
            33,
            f"Max {maxkey}  ={max_temp:.1f}ºC",
            rotation=0,
            color=temp_colors__wid[maxkey][0],
        )

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        # save to .png
        plt.tight_layout()
        dest_fn = os.path.join(run_dir, run_bit + '.png')
        plt.savefig(
            dest_fn,
            dpi=300,
        )

        plt.show()

    def a2seran_bigbox_hotair2(self, csv_fn='~/Downloads/pyrgepyran/2seran_bigbox_hotair2/2seran_bigbox_hotair2.csv'):
        import matplotlib.pyplot as plt

        title = "05 June 2024; Big Box, 2 serans, hot air on/off"

        csv_fn = os.path.expanduser(csv_fn)
        run_dir = os.path.dirname(csv_fn)
        run_bit = os.path.basename(csv_fn).rsplit('.', 1)[0]

        normal_line_width = 2
        emphasis_line_width = 3

        baseline_env_Wm2 = 540

        temp_mappings = {
            'No.5': 'inside_air_C',
            'No.6': '3rd_layer_C',
            'No.7': '2nd_layer_C',
            'No.8': 'top_layer_uppermid_C',
        }

        logger_temp_mappings = {
            # '1ch': 'NONE',
            '1ch': 'seran3_layer_C',
            '2ch': 'bigbox_inside_bottom_C',
            '3ch': 'bigbox_inside_top_C',
        }

        df = load_and_process_csv(
            csv_fn,
            temp_mappings=temp_mappings,
            baseline_env_Wm2=baseline_env_Wm2,
            offset_secs=0,
        )
        df_logger = load_and_process_logger_txts(
            run_dir,
            temp_mappings=logger_temp_mappings,
            offset_secs=0,
        )

        # # drop anything before 16:00
        df = df[df['Date/Time'] > pd.to_datetime('05 June 2024, 16:05')]
        df_logger = df_logger[df_logger['Date/Time'] > pd.to_datetime('05 June 2024, 16:05')]

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(12, 10),
        )

        ax.set_title(title)

        to_plot = df['solar_in_Wm2']
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # make a second left plot for just netir
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', -0.15))

        ax3.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax3.get_lines()[-1].set_color('red')
        ax3.get_lines()[-1].set_linewidth(normal_line_width)

        # make a 3rd left plot for total
        ax4 = ax.twinx()
        ax4.spines['right'].set_position(('axes', -0.25))

        # ax4.plot(df['Date/Time'], df['total_in_Wm2'], label='Solar + IR in')
        # ax4.get_lines()[-1].set_color('gray')
        # ax4.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM:SS
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM:SS)')

        # y axis label: W/m^2
        ax.set_ylabel('Solar (W/m^2)')
        ax3.set_ylabel('Net IR (W/m^2)')
        ax4.set_ylabel('Solar + IR In (W/m^2)')

        ax1_wm2_min = 400
        ax1_wm2_max = 800
        ax.set_ylim(ax1_wm2_min, ax1_wm2_max)
        ax1_wm2_range = ax1_wm2_max - ax1_wm2_min
        # set y ticks every wm2_range / 8
        ax.set_yticks(
            [bit*(ax1_wm2_range/8) + ax1_wm2_min for bit in range(8+1)]
        )

        ax3_wm2_min = -300
        ax3_wm2_max = 100
        ax3.set_ylim(ax3_wm2_min, ax3_wm2_max)
        ax3_wm2_range = ax3_wm2_max - ax3_wm2_min
        # set y ticks every wm2_range / 8
        ax3.set_yticks(
            [bit*(ax3_wm2_range/8) + ax3_wm2_min for bit in range(8+1)]
        )

        ax4_wm2_min = 1000
        ax4_wm2_max = 1400
        ax4.set_ylim(ax4_wm2_min, ax4_wm2_max)
        ax4_wm2_range = ax4_wm2_max - ax4_wm2_min
        # set y ticks every wm2_range / 8
        ax4.set_yticks(
            [bit*(ax4_wm2_range/8) + ax4_wm2_min for bit in range(8+1)]
        )

        # # draw red horizontal line at 0
        # ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # make a 2nd y-axis on ax plot
        ax2 = ax.twinx()

        temp_colors__wid = {
            'thermistor_T_C': ('#ff33ff', normal_line_width),
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
            '3rd_layer_C': ('gray', normal_line_width),
            # 'No.7': '2nd_layer_C',
            '2nd_layer_C': ('brown', normal_line_width),
            'top_layer_C': ('#4F0000', emphasis_line_width),
            'top_layer_uppermid_C': ('#4F0000', normal_line_width),
            'seran2_air_C': ('#7777ff', normal_line_width),
            'seran2_layer_C': ('#3333ff', normal_line_width),
            'seran3_air_C': ('#2222ff', normal_line_width),
            'seran3_layer_C': ('#0000bf', normal_line_width),
            'top_layer_lowermid_C': ('#FF0000', normal_line_width),
            'top_layer_lowerleft_C': ('#FF7777', normal_line_width),

            'bigbox_inside_bottom_C': ('#88cc88', normal_line_width),
            'bigbox_inside_top_C': ('#bbffbb', normal_line_width),
        }
        for temp_key in temp_colors__wid:
            # if not in df, skip
            if temp_key not in df:
                continue

            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        if df_logger is not None:
            for temp_key in temp_colors__wid:
                # if not in df, skip
                if temp_key not in df_logger:
                    continue

                col, wid = temp_colors__wid[temp_key]
                ax2.plot(df_logger['Date/Time'], df_logger[temp_key], label=temp_key, color=col)
                ax2.get_lines()[-1].set_linewidth(wid)

        # legend bottom-left
        ax2.legend(title='Temperature', loc='upper right')
        ax2.set_xlabel('Time (HH:MM:SS)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax2.set_ylabel('Temperature (ºC)')

        cymin = 30
        cymax = 130
        cyrange = cymax - cymin
        ax2.set_ylim(cymin, cymax)
        # set 8th ticks
        ax2.set_yticks(
            [bit*(cyrange/8) + cymin for bit in range(8+1)]
        )

        plt.setp(ax2.get_xticklabels(), rotation=45)
        #

        for adj_time, color, reason, linewidth in [
            ('05 Jun 2024, 16:13:06', 'darkgreen', '', 1),
            ('05 Jun 2024, 16:13:22', 'darkorange', '', 1),
            ('05 Jun 2024, 16:13:24', 'darkred', '', 1),
            ('05 Jun 2024, 16:13:42', 'orange', '', 1),

            ('05 Jun 2024, 16:40:11', 'darkgreen', '', 1),
            ('05 Jun 2024, 16:40:30', 'darkorange', '', 1),
            ('05 Jun 2024, 16:41:15', 'darkred', '', 1),
            ('05 Jun 2024, 16:41:36', 'orange', '', 1),

            ('05 Jun 2024, 16:46:10', 'darkgreen', '', 1),
            ('05 Jun 2024, 16:46:26', 'darkorange', '', 1),
            ('05 Jun 2024, 16:47:01', 'darkred', '', 1),
            ('05 Jun 2024, 16:47:07', 'orange', '', 1),

            ('05 Jun 2024, 16:48:55', 'darkgreen', '', 1),
            ('05 Jun 2024, 16:49:02', 'darkorange', '', 1),
            ('05 Jun 2024, 16:50:15', 'darkred', '', 1),
            ('05 Jun 2024, 16:50:21', 'orange', '', 1),

            ('05 Jun 2024, 16:51:29', 'darkgreen', '', 1),
            ('05 Jun 2024, 16:51:47', 'darkorange', '', 1),
            ('05 Jun 2024, 16:52:54', 'darkred', '', 1),
            ('05 Jun 2024, 16:53:12', 'orange', '', 1),

            ('05 Jun 2024, 16:58:26', 'darkgreen', '', 1),
            ('05 Jun 2024, 16:58:42', 'darkorange', '', 1),
            ('05 Jun 2024, 16:59:39', 'darkred', '', 1),
            ('05 Jun 2024, 16:59:56', 'orange', '', 1),
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=linewidth)

            # draw label of the reason on top graph
            ax.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=5),
                1000,
                reason,
                rotation=0,
                color=color,
            )

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        # place ax legend on top-left and ax2 on top-right
        ax.legend(title='Power', loc='upper left')
        ax2.legend(title='Temperature', loc='upper right')

        # save to .png
        plt.tight_layout()
        dest_fn = os.path.join(run_dir, run_bit + '.png')
        plt.savefig(
            dest_fn,
            dpi=300,
        )

        plt.show()

    def a2seran_bigbox_hotair3(self, csv_fn='~/Downloads/pyrgepyran/2seran_bigbox_hotair3/2seran_bigbox_hotair3.csv'):
        import matplotlib.pyplot as plt

        title = "05 July 2024; Big Box, 2 serans, hot air` on/off"

        csv_fn = os.path.expanduser(csv_fn)
        run_dir = os.path.dirname(csv_fn)
        run_bit = os.path.basename(csv_fn).rsplit('.', 1)[0]

        normal_line_width = 1
        emphasis_line_width = 2

        baseline_env_Wm2 = 540

        temp_mappings = {
            'No.5': 'top_seran_C',
            'No.6': 'air_between_C',
            'No.7': 'bottom_inside_air_C',
            'No.8': 'black_plate_C',
        }

        logger_temp_mappings = {
            # '1ch': 'NONE',
            '1ch': 'lowest_seran_C',
            '2ch': 'lowest_seran_air_C',
            '3ch': 'top_inside_air_C',
        }

        df = load_and_process_csv(
            csv_fn,
            temp_mappings=temp_mappings,
            baseline_env_Wm2=baseline_env_Wm2,
            offset_secs=0,
        )
        df_logger = load_and_process_logger_txts(
            run_dir,
            temp_mappings=logger_temp_mappings,
            offset_secs=0,
        )

        # # add offsets
        # df['bottom_inside_air_C'] += 9
        # df_logger['lowest_seran_air_C'] += 28

        # # # drop anything before 16:00
        # df = df[df['Date/Time'] > pd.to_datetime('05 June 2024, 16:05')]
        # df_logger = df_logger[df_logger['Date/Time'] > pd.to_datetime('05 June 2024, 16:05')]

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(12, 10),
        )

        ax.set_title(title)

        to_plot = df['solar_in_Wm2']
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # make a second left plot for just netir
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', -0.15))

        ax3.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        ax3.get_lines()[-1].set_color('red')
        ax3.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM:SS
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM:SS)')

        # y axis label: W/m^2
        ax.set_ylabel('Solar (W/m^2)')
        ax3.set_ylabel('Net IR (W/m^2)')

        ax1_wm2_min = 200
        ax1_wm2_max = 600
        ax.set_ylim(ax1_wm2_min, ax1_wm2_max)
        ax1_wm2_range = ax1_wm2_max - ax1_wm2_min
        # set y ticks every wm2_range / 8
        ax.set_yticks(
            [bit*(ax1_wm2_range/8) + ax1_wm2_min for bit in range(8+1)]
        )

        ax3_wm2_min = -300
        ax3_wm2_max = 100
        ax3.set_ylim(ax3_wm2_min, ax3_wm2_max)
        ax3_wm2_range = ax3_wm2_max - ax3_wm2_min
        # set y ticks every wm2_range / 8
        ax3.set_yticks(
            [bit*(ax3_wm2_range/8) + ax3_wm2_min for bit in range(8+1)]
        )

        # # make a 3rd left plot for total
        # ax4 = ax.twinx()
        # ax4.spines['right'].set_position(('axes', -0.25))

        # ax4.plot(df['Date/Time'], df['total_in_Wm2'], label='Solar + IR in')
        # ax4.get_lines()[-1].set_color('gray')
        # ax4.get_lines()[-1].set_linewidth(normal_line_width)

        # ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        # ax4.set_ylabel('Solar + IR In (W/m^2)')
        # ax4_wm2_min = 1000
        # ax4_wm2_max = 1400
        # ax4.set_ylim(ax4_wm2_min, ax4_wm2_max)
        # ax4_wm2_range = ax4_wm2_max - ax4_wm2_min
        # # set y ticks every wm2_range / 8
        # ax4.set_yticks(
        #     [bit*(ax4_wm2_range/8) + ax4_wm2_min for bit in range(8+1)]
        # )

        # # draw red horizontal line at 0
        # ax.axhline(0, color='red', linestyle='--', linewidth=1)

        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # make a 2nd y-axis on ax plot
        ax2 = ax.twinx()

        temp_colors__wid = {
            'thermistor_T_C': ('#ff33ff', normal_line_width),
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
            '3rd_layer_C': ('gray', normal_line_width),
            # 'No.7': '2nd_layer_C',
            '2nd_layer_C': ('brown', normal_line_width),
            'top_layer_C': ('#4F0000', emphasis_line_width),
            'top_layer_uppermid_C': ('#4F0000', normal_line_width),
            'seran2_air_C': ('#7777ff', normal_line_width),
            'seran2_layer_C': ('#3333ff', normal_line_width),
            'seran3_air_C': ('#2222ff', normal_line_width),
            'seran3_layer_C': ('#0000bf', normal_line_width),
            'top_layer_lowermid_C': ('#FF0000', normal_line_width),
            'top_layer_lowerleft_C': ('#FF7777', normal_line_width),

            'top_inside_air_C': ('#6ADEF5', normal_line_width),
            'top_seran_C': ('green', normal_line_width),
            'air_between_C': ('blue', normal_line_width),
            'bottom_inside_air_C': ('#327583', normal_line_width),
            'black_plate_C': ('black', normal_line_width),
            'lowest_seran_C': ('orange', normal_line_width),
            'lowest_seran_air_C': ('purple', normal_line_width),
        }
        for temp_key in temp_colors__wid:
            # if not in df, skip
            if temp_key not in df:
                continue

            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        if df_logger is not None:
            for temp_key in temp_colors__wid:
                # if not in df, skip
                if temp_key not in df_logger:
                    continue

                col, wid = temp_colors__wid[temp_key]
                ax2.plot(df_logger['Date/Time'], df_logger[temp_key], label=temp_key, color=col)
                ax2.get_lines()[-1].set_linewidth(wid)

        ax2.set_xlabel('Time (HH:MM:SS)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax2.set_ylabel('Temperature (ºC)')

        cymin = 25
        cymax = 110
        cyrange = cymax - cymin
        ax2.set_ylim(cymin, cymax)
        # set 8th ticks
        ax2.set_yticks(
            [bit*(cyrange/8) + cymin for bit in range(8+1)]
        )

        plt.setp(ax2.get_xticklabels(), rotation=45)
        #

        for adj_time, color, reason, linewidth in [
            ('05 Jul 2024, 16:23:48', 'orange', '', 1),
            ('05 Jul 2024, 16:24:48', 'red', '', 1),
            ('05 Jul 2024, 16:25:49', 'orange', '', 1),
            ('05 Jul 2024, 16:26:49', 'red', '', 1),
            ('05 Jul 2024, 16:27:50', 'orange', '', 1),
            ('05 Jul 2024, 16:28:50', 'red', '', 1),

            ('05 Jul 2024, 16:29:49', 'orange', '', 1),
            ('05 Jul 2024, 16:30:49', 'red', '', 1),
            ('05 Jul 2024, 16:31:50', 'orange', '', 1),
            ('05 Jul 2024, 16:32:50', 'red', '', 1),
            ('05 Jul 2024, 16:33:50', 'orange', '', 1),
            ('05 Jul 2024, 16:34:50', 'red', '', 1),

            ('05 Jul 2024, 16:49:40', 'orange', '', 1),
            ('05 Jul 2024, 16:50:40', 'red', '', 1),
            ('05 Jul 2024, 16:51:41', 'orange', '', 1),
            ('05 Jul 2024, 16:52:41', 'red', '', 1),
            ('05 Jul 2024, 16:53:42', 'orange', '', 1),
            ('05 Jul 2024, 16:54:42', 'red', '', 1),

            ('05 Jul 2024, 16:55:45', 'orange', '', 1),
            ('05 Jul 2024, 16:56:45', 'red', '', 1),
            ('05 Jul 2024, 16:57:45', 'orange', '', 1),
            ('05 Jul 2024, 16:58:45', 'red', '', 1),
            ('05 Jul 2024, 16:59:46', 'orange', '', 1),
            ('05 Jul 2024, 17:00:46', 'red', '', 1),

            # solar dips
            ('05 Jul 2024, 16:36:08', 'yellow', '', 1),
            ('05 Jul 2024, 16:36:20', 'yellow', '', 1),
            ('05 Jul 2024, 16:38:21', 'yellow', '', 1),
            ('05 Jul 2024, 16:40:21', 'yellow', '', 1),

            # # when lowest seran air C increases
            # ('05 Jul 2024, 16:30:13', 'purple', '', 1),
            # ('05 Jul 2024, 16:32:26', 'purple', '', 1),
            # ('05 Jul 2024, 16:34:20', 'purple', '', 1),
            #
            # # when bottom_inside_air_C slope starts to increase
            # ('05 Jul 2024, 16:29:58', '#327583', '', 1),
            # ('05 Jul 2024, 16:32:07', '#327583', '', 1),
            # ('05 Jul 2024, 16:33:59', '#327583', '', 1),
            #
            # # when the black plate C slope starts to increase
            # ('05 Jul 2024, 16:29:57', 'black', '', 1),
            # ('05 Jul 2024, 16:31:55', 'black', '', 1),
            # ('05 Jul 2024, 16:33:52', 'black', '', 1),

        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=linewidth)

            # draw label of the reason on top graph
            ax.text(
                pd.to_datetime(adj_time) + pd.Timedelta(seconds=5),
                1000,
                reason,
                rotation=0,
                color=color,
            )

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)

        # place ax legend on top-left and ax2 on top-right
        ax.legend(title='', loc='upper left')
        ax3.legend(title='', loc='upper left', bbox_to_anchor=(0, 0.975))
        ax2.legend(title='Temperature', loc='upper right')

        # save to .png
        plt.tight_layout()
        dest_fn = os.path.join(run_dir, run_bit + '.png')
        plt.savefig(
            dest_fn,
            dpi=300,
        )

        plt.show()

    def a2seran_bigbox_hotair4(self, csv_fn='~/Downloads/pyrgepyran/2seran_bigbox_hotair4/2seran_bigbox_hotair4.csv'):
        import matplotlib.pyplot as plt

        title = "06 July 2024; Big Box, 2 serans, hot air on/off"

        csv_fn = os.path.expanduser(csv_fn)
        run_dir = os.path.dirname(csv_fn)
        run_bit = os.path.basename(csv_fn).rsplit('.', 1)[0]

        normal_line_width = 1
        emphasis_line_width = 2

        baseline_env_Wm2 = 540

        temp_mappings = {
            'No.5': 'top_seran_C',
            'No.6': 'air_between_C',
            'No.7': 'bottom_inside_air_C',
            'No.8': 'black_plate_C',
        }

        logger_temp_mappings = {
            # '1ch': 'NONE',
            '1ch': 'lowest_seran_C',
            '2ch': 'middle_seran_C',
            '3ch': 'top_inside_air_C',
        }

        df = load_and_process_csv(
            csv_fn,
            temp_mappings=temp_mappings,
            baseline_env_Wm2=baseline_env_Wm2,
            offset_secs=0,
        )
        df_logger = load_and_process_logger_txts(
            run_dir,
            temp_mappings=logger_temp_mappings,
            offset_secs=0,
        )

        # # add offsets
        # df['bottom_inside_air_C'] += 9
        # df_logger['lowest_seran_air_C'] += 28

        # # drop anything before x time
        df = df[df['Date/Time'] > pd.to_datetime('06 July 2024, 15:16')]
        df_logger = df_logger[df_logger['Date/Time'] > pd.to_datetime('06 July 2024, 15:16')]

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(12, 10),
        )

        ax.set_title(title)

        to_plot = df['solar_in_Wm2']
        ax.plot(df['Date/Time'], to_plot, label='Solar')
        ax.get_lines()[-1].set_color('#DFDF00')
        ax.get_lines()[-1].set_linewidth(normal_line_width)

        # make a second left plot for just netir
        ax3 = ax.twinx()
        ax3.spines['right'].set_position(('axes', -0.15))

        # ax3.plot(df['Date/Time'], df['ir_net_Wm2'], label='Net IR')
        # ax3.get_lines()[-1].set_color('red')
        # ax3.get_lines()[-1].set_linewidth(normal_line_width)

        ax3.plot(df['Date/Time'], df['ir_in_Wm2'], label='IR In')
        ax3.get_lines()[-1].set_color('red')
        ax3.get_lines()[-1].set_linewidth(normal_line_width)

        # ax.legend(title='Power', loc='upper right')

        # set x axis formatter to HH:MM:SS
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax3.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM:SS)')

        # y axis label: W/m^2
        ax.set_ylabel('Solar (W/m^2)')
        ax3.set_ylabel('IR (W/m^2)')

        ax1_wm2_min = 300
        ax1_wm2_max = 700
        ax.set_ylim(ax1_wm2_min, ax1_wm2_max)
        ax1_wm2_range = ax1_wm2_max - ax1_wm2_min
        # set y ticks every wm2_range / 8
        ax.set_yticks(
            [bit*(ax1_wm2_range/8) + ax1_wm2_min for bit in range(8+1)]
        )

        ax3_wm2_min = 500
        ax3_wm2_max = 900
        ax3.set_ylim(ax3_wm2_min, ax3_wm2_max)
        ax3_wm2_range = ax3_wm2_max - ax3_wm2_min
        # set y ticks every wm2_range / 8
        ax3.set_yticks(
            [bit*(ax3_wm2_range/8) + ax3_wm2_min for bit in range(8+1)]
        )

        # set minor ticks within /40 of range
        ax3.set_yticks(
            [bit*(ax3_wm2_range/40) + ax3_wm2_min for bit in range(40+1)],
            minor=True,
        )

        # draw red horizontal line at 0
        ax3.axhline(0, color='red', linestyle='--', linewidth=1)

        # # make a 3rd left plot for total
        # ax4 = ax.twinx()
        # ax4.spines['right'].set_position(('axes', -0.25))

        # ax4.plot(df['Date/Time'], df['total_in_Wm2'], label='Solar + IR in')
        # ax4.get_lines()[-1].set_color('gray')
        # ax4.get_lines()[-1].set_linewidth(normal_line_width)

        # ax4.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        # ax4.set_ylabel('Solar + IR In (W/m^2)')
        # ax4_wm2_min = 1000
        # ax4_wm2_max = 1400
        # ax4.set_ylim(ax4_wm2_min, ax4_wm2_max)
        # ax4_wm2_range = ax4_wm2_max - ax4_wm2_min
        # # set y ticks every wm2_range / 8
        # ax4.set_yticks(
        #     [bit*(ax4_wm2_range/8) + ax4_wm2_min for bit in range(8+1)]
        # )


        # rotate x axis labels
        plt.setp(ax.get_xticklabels(), rotation=45)

        # make a 2nd y-axis on ax plot
        ax2 = ax.twinx()

        temp_colors__wid = {
            'thermistor_T_C': ('#ff33ff', normal_line_width),
            'inside_air_C': ('cyan', normal_line_width),
            'aluminum_C': ('gray', normal_line_width),
            'styro_5mm_C': ('brown', normal_line_width),
            'styro_surf_C': ('#4F0000', emphasis_line_width),
            '3rd_layer_C': ('gray', normal_line_width),
            # 'No.7': '2nd_layer_C',
            '2nd_layer_C': ('brown', normal_line_width),
            'top_layer_C': ('#4F0000', emphasis_line_width),
            'top_layer_uppermid_C': ('#4F0000', normal_line_width),
            'seran2_air_C': ('#7777ff', normal_line_width),
            'seran2_layer_C': ('#3333ff', normal_line_width),
            'seran3_air_C': ('#2222ff', normal_line_width),
            'seran3_layer_C': ('#0000bf', normal_line_width),
            'top_layer_lowermid_C': ('#FF0000', normal_line_width),
            'top_layer_lowerleft_C': ('#FF7777', normal_line_width),

            'top_inside_air_C': ('#6ADEF5', normal_line_width),
            'top_seran_C': ('green', normal_line_width),
            'air_between_C': ('blue', normal_line_width),
            'bottom_inside_air_C': ('#327583', normal_line_width),
            'black_plate_C': ('black', normal_line_width),
            'lowest_seran_C': ('orange', normal_line_width),
            'middle_seran_C': ('darkred', normal_line_width),
            'lowest_seran_air_C': ('purple', normal_line_width),
        }
        for temp_key in temp_colors__wid:
            # if not in df, skip
            if temp_key not in df:
                continue

            col, wid = temp_colors__wid[temp_key]
            ax2.plot(df['Date/Time'], df[temp_key], label=temp_key, color=col)
            ax2.get_lines()[-1].set_linewidth(wid)

        if df_logger is not None:
            for temp_key in temp_colors__wid:
                # if not in df, skip
                if temp_key not in df_logger:
                    continue

                col, wid = temp_colors__wid[temp_key]
                ax2.plot(df_logger['Date/Time'], df_logger[temp_key], label=temp_key, color=col)
                ax2.get_lines()[-1].set_linewidth(wid)

        ax2.set_xlabel('Time (HH:MM:SS)')
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M:%S'))
        ax2.set_ylabel('Temperature (ºC)')

        cymin = 20
        cymax = 100
        cyrange = cymax - cymin
        ax2.set_ylim(cymin, cymax)
        # set 8th ticks
        ax2.set_yticks(
            [bit*(cyrange/8) + cymin for bit in range(8+1)]
        )

        plt.setp(ax2.get_xticklabels(), rotation=45)
        #

        for adj_time, color, reason, linewidth in [
            ('06 Jul 2024, 15:17:35', 'orange', '', 1),
            ('06 Jul 2024, 15:19:35', 'red', '', 1),
            ('06 Jul 2024, 15:24:54', 'orange', '', 1),
            ('06 Jul 2024, 15:27:54', 'red', '', 1),
            ('06 Jul 2024, 15:34:56', 'orange', '', 1),
            ('06 Jul 2024, 15:37:56', 'red', '', 1),
            ('06 Jul 2024, 15:42:56', 'orange', '', 1),
            ('06 Jul 2024, 15:45:56', 'red', '', 1),
            ('06 Jul 2024, 15:50:56', 'orange', '', 1),
            ('06 Jul 2024, 15:53:56', 'red', '', 1),
            ('06 Jul 2024, 15:58:56', 'orange', '', 1),
            ('06 Jul 2024, 16:01:56', 'red', '', 1),
            ('06 Jul 2024, 16:06:56', 'orange', '', 1),
            ('06 Jul 2024, 16:09:56', 'red', '', 1),

            ('06 Jul 2024, 16:05:13', 'purple', 'adjust', 1),

            # set one to expand graph
            ('06 Jul 2024, 16:20:00', 'white', '', 1),
        ]:
            for a in [ax, ax2]:
                a.axvline(pd.to_datetime(adj_time), color=color, linestyle='--', linewidth=linewidth)

            # draw label of the reason on top graph
            if reason:
                ax.text(
                    pd.to_datetime(adj_time) + pd.Timedelta(seconds=5),
                    670,
                    reason,
                    rotation=0,
                    color=color,
                )

        ax.grid(axis='y', linestyle='-', linewidth=0.5)
        ax2.grid(axis='y', linestyle='-', linewidth=0.5)
        # set minor grid on ax3
        ax3.grid(axis='y', linestyle='--', linewidth=0.5, which='minor')

        # place ax legend on top-left and ax2 on top-right
        ax.legend(title='', loc='upper left')
        ax3.legend(title='', loc='upper left', bbox_to_anchor=(0, 0.975))
        ax2.legend(title='Temperature', loc='upper right')

        # save to .png
        plt.tight_layout()
        dest_fn = os.path.join(run_dir, run_bit + '.png')
        plt.savefig(
            dest_fn,
            dpi=300,
        )

        plt.show()


if __name__ == '__main__':
    fire.Fire(CmdLine)
