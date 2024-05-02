import numpy as np
import pandas as pd
import subprocess
import fire
import math


def load_and_process_csv(csv_fn, temp_mappings=None):
    # load up the csv file
    df = pd.read_csv(csv_fn, encoding='utf8')

    # drop 1st 3 rows
    df = df[3:]

    # drop second "Date/Time" column
    df = df.drop(columns=['Date/Time.1'])

    # convert "Date/Time" col data to timestamp
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

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

    # calculate thermistor T:
    excitation_voltage = np.array(df['exc_V'], dtype=np.float64)
    Vr = np.array(df['thermistor_V'], dtype=np.float64) / excitation_voltage
    Rt = 24900 * (Vr/(1 - Vr))
    A = 9.32794e-4
    B = 2.21451e-4
    C = 1.26233e-7
    thermistor_T_K = 1 / (A + B * np.log(Rt) + C * ((np.log(Rt))**3))
    df['thermistor_T_C'] = thermistor_T_K - 273.15

    pyrg_k1 = 8.986
    pyrg_k2 = 1.028
    sb = 5.6704e-8
    ir_net_V = np.array(df['ir_net_V'], dtype=np.float64)
    df['ir_net_Wm2'] = ir_net_Wm2 = pyrg_k1 * (ir_net_V*1000)
    df['ir_in_Wm2'] = ir_in_Wm2 = pyrg_k1 * (ir_net_V*1000) + pyrg_k2 * sb * thermistor_T_K**4
    df['ir_out_Wm2'] = pyrg_k2 * sb * thermistor_T_K**4
    df['total_in_Wm2'] = solar_in_Wm2 + ir_in_Wm2

    df['max_implied_solar_C'] = (
        ((solar_in_Wm2) / sb)**(1/4) - 273.15
    )

    df['max_implied_total_C'] = (
        ((solar_in_Wm2 + ir_in_Wm2) / sb)**(1/4) - 273.15
    )

    # the conductive/convective loss must be equal to solar+ir
    df['cond_conv_loss_Wm2'] = solar_in_Wm2 + ir_net_Wm2

    return df


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

        # drop anything before 13:00
        df = df[df['Date/Time'] > pd.to_datetime('13:00')]

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


if __name__ == '__main__':
    fire.Fire(CmdLine)
