import numpy as np
import pandas as pd
import subprocess
import fire
import math


def load_and_process_csv(csv_fn):
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
        'No.4': 'tabletop_C',
    }
    df = df.rename(columns=rename)

    df['tabletop_C'] = np.array(df['tabletop_C'], dtype=np.float64)

    # remove rows with any nan values
    df = df.dropna()

    pyran_k1 = 24.51

    # add column which is this k1 * the solar in V
    solar_in_V = np.array(df['solar_in_V'], dtype=np.float64)
    df['solar_in_Wm2'] = solar_in_Wm2 = pyran_k1 * (solar_in_V*1000)

    # calculate thermistor T:
    excitation_voltage = 2.5
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
    df['ir_net_Wm2'] = pyrg_k1 * (ir_net_V*1000)
    df['ir_in_Wm2'] = ir_in_Wm2 = pyrg_k1 * (ir_net_V*1000) + pyrg_k2 * sb * thermistor_T_K**4
    df['ir_out_Wm2'] = pyrg_k2 * sb * thermistor_T_K**4
    df['total_in_Wm2'] = solar_in_Wm2 + ir_in_Wm2

    df['max_implied_solar_C'] = (
        ((solar_in_Wm2) / sb)**(1/4) - 273.15
    )

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

    def plot_raw(self, csv_fn):
        # load it up to pandas dataframe
        df = load_and_process_csv(csv_fn)

        # export as xls
        df.to_excel('output.xlsx', index=False)

        # plot solar_in_Wm2 and ir_in_Wm2 on y axiw tih date on x
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        ax.plot(df['Date/Time'], df['solar_in_Wm2'], label='Solar')
        ax.plot(df['Date/Time'], df['ir_in_Wm2'], label='IR (in)')
        ax.plot(df['Date/Time'], df['ir_out_Wm2'], label='IR (out)')
        ax.plot(df['Date/Time'], df['total_in_Wm2'], label='Total in')
        # set solar plot color to dark yellow
        ax.get_lines()[0].set_color('goldenrod')
        # IR to dark red
        ax.get_lines()[1].set_color('darkred')
        # IR out to dark purple
        ax.get_lines()[2].set_color('indigo')
        # Total in to dark green
        ax.get_lines()[3].set_color('darkgreen')
        # legend upper in the center
        ax.legend(title='Power', loc='upper center')

        # set x axis formatter to HH:MM
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%H:%M'))

        # x axis label: time (HH:MM)
        ax.set_xlabel('Time (HH:MM)')

        # y axis label: W/m^2
        ax.set_ylabel('Power/Area (W/m^2)')

        # rotate x axis labels
        plt.xticks(rotation=45)

        # on a separate Y axis, plot the thermistor_T_C
        ax2 = ax.twinx()
        ax2.plot(df['Date/Time'], df['thermistor_T_C'], color='red', label='Thermistor')
        ax2.plot(df['Date/Time'], df['tabletop_C'], color='blue', label='Tabletop')
        ax2.plot(df['Date/Time'], df['max_implied_solar_C'], color='black', linestyle='dotted', label='Max Implied Solar')
        # thinner lines
        ax2.get_lines()[0].set_linewidth(0.5)
        ax2.get_lines()[1].set_linewidth(0.5)
        # legend bottom-left
        ax2.legend(title='Temperature', loc='lower left')
        ax2.set_ylabel('Temperature (ºC)')
        # y axis: min 0, max 100
        ax2.set_ylim(0, 100)

        plt.show()



if __name__ == '__main__':
    fire.Fire(CmdLine)
