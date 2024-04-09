import tabulate
import pandas as pd
import fire
import chardet
import os
import subprocess


def load_and_process_csv(csv_fn):
    # load up the csv file
    df = pd.read_csv(csv_fn, encoding='utf8')

    # drop 1st 3 rows
    df = df[3:]

    # convert "Date/Time" col data to timestamp
    df['Date/Time'] = pd.to_datetime(df['Date/Time'])

    # drop Date/time2 column
    df = df.drop(columns=['Date/Time2'])

    # convert other columns to float
    for col in df.columns[1:]:
        if col == 'Date/Time':
            continue
        df[col] = df[col].str.replace(',', '').astype(float)

    # multiply HPVoltage by 1.835
    amps = None
    if 'HPVoltage' in df.columns:
        # split fn parts by _ and see if any ends with "A"
        parts = os.path.basename(csv_fn).split('_')
        amps = None
        for part in parts:
            if part.endswith('A'):
                amps = float(part[:-1])
                break
        else:
            amps = float(input("What amperage? "))
        print("Using amperage %.3fA" % amps)
        df['HPVoltage'] *= amps

        # and rename it to HPPower
        df = df.rename(columns={'HPVoltage': 'HPPower'})

    # add column Room<->Bottom, which is InsideBottom - RoomAir
    if 'RoomAir' in df.columns and 'InsideBottom' in df.columns:
        df['Room<->Bottom'] = df['InsideBottom'] - df['RoomAir']

    # add column Bottom<->Top, which is InsideBottom - InsideTop
    if 'InsideTop' in df.columns and 'InsideBottom' in df.columns:
        df['Bottom<->Top'] = df['InsideBottom'] - df['InsideTop']

    # add column Bottom<->Mid which is InsideBottom - InsideMidAir
    if 'InsideMidAir' in df.columns and 'InsideBottom' in df.columns:
        df['Bottom<->Mid'] = df['InsideBottom'] - df['InsideMidAir']

    # drop No.1 to No.4 cols, if they exst
    for col in ['No.1', 'No.2', 'No.3', 'No.4']:
        if col in df.columns:
            df = df.drop(columns=[col])

    return df, amps


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

    def thirtymin_chunks(self, csv_fn):
        df, amps = load_and_process_csv(csv_fn)

        # split into 30 min chunks
        start_ts = df['Date/Time'].iloc[0]
        print(start_ts)
        # snap to 00 or 30
        start_ts = start_ts.replace(minute=0 if start_ts.minute < 30 else 30, second=0, microsecond=0)
        print(start_ts)

        new_table = []

        while True:
            end_ts = start_ts + pd.Timedelta(minutes=30)

            # for each column, get mean and stddev in range and print
            print("From", start_ts, "to", end_ts)
            new_row = []
            new_row.append("%s to %s" % (start_ts, end_ts))
            for col in df.columns[1:]:
                mean = df[(df['Date/Time'] >= start_ts) & (df['Date/Time'] < end_ts)][col].mean()
                stddev = df[(df['Date/Time'] >= start_ts) & (df['Date/Time'] < end_ts)][col].std()
                print(f"{col:10s}: {mean:.2f} +/- {stddev:.2f}")
                # for power use 4 decimals, otherwise use 2
                if col == 'HPPower':
                    new_row.append("%.3f +/- %.3f" % (mean, 2*stddev))
                else:
                    new_row.append("%.2f +/- %.2f" % (mean, 2*stddev))

            new_table.append(new_row)
            print()
            start_ts = end_ts
            if start_ts > df['Date/Time'].iloc[-1]:
                break

        res = tabulate.tabulate(new_table, headers=['Time', *df.columns[1:]], tablefmt='pretty')
        print(res)

        if amps:
            print("Amperage used: %.3fA" % amps)

    def steady_state_summary(self, csv_fn, start_time_str, end_time_str):
        df, amps = load_and_process_csv(csv_fn)

        # parse time strings to datetime
        start_time = pd.to_datetime(start_time_str)
        end_time = pd.to_datetime(end_time_str)
        print("-------------------------------")
        print("Ranges:")
        # print min&max values of each like:
        # ColName - min - max
        for col in df.columns[1:]:
            minval = df[(df['Date/Time'] >= start_time) & (df['Date/Time'] < end_time)][col].min()
            maxval = df[(df['Date/Time'] >= start_time) & (df['Date/Time'] < end_time)][col].max()
            # use %.3f for power, otherwise %.1f
            if col == 'HPPower':
                print("%-15s: %.3f - %.3f" % (col, minval, maxval))
            else:
                print("%-15s: %.1f - %.1f" % (col, minval, maxval))

        # drop all but <-> cols and Date/Time and HPPower
        df = df[['Date/Time', 'HPPower'] + [col for col in df.columns if '<->' in col]]
        new_row = []
        new_row.append("%s to %s" % (start_time, end_time))
        for col in df.columns[1:]:
            mean = df[(df['Date/Time'] >= start_time) & (df['Date/Time'] < end_time)][col].mean()
            stddev = df[(df['Date/Time'] >= start_time) & (df['Date/Time'] < end_time)][col].std()
            # for power use 4 decimals, otherwise use 2
            if col == 'HPPower':
                new_row.append("%.3f +/- %.3f" % (mean, 2*stddev))
            else:
                new_row.append("%.2f +/- %.2f" % (mean, 2*stddev))

        res = tabulate.tabulate([new_row], headers=['Time', *df.columns[1:]], tablefmt='pretty')
        print(res)

        # print min and m

    def twoway_rad_solver(self, T1, T2, e1, e2, f12, A1, A2):
        from scipy.optimize import least_squares
        sb = 5.670374419e-8

        print(T1, T2, e1, e2, f12, A1, A2)

        Eb1 = sb * T1 ** 4
        Eb2 = sb * T2 ** 4

        R1 = (1 - e1) / (A1 * e1)
        R2 = (1 - e2) / (A2 * e2)
        R12 = 1 / (A1 * f12)

        print(Eb1 - Eb2)
        print(R1, R12, R2)

        Q1 = (Eb1 - Eb2) / (R1 + R12 + R2)
        print("Heat flow at surf 1:", Q1)
        return Q1

        # def equations(vars):
        #     J1, J2 = vars
        #
        #     eqs = [
        #         (Eb1 - J1) / R1 + (J2 - J1) / R12,
        #         (Eb2 - J2) / R2 + (J1 - J2) / R12,
        #     ]
        #     return eqs
        #
        # # Initial guess for J1, J2
        # initial_guess = [1, 1]
        #
        # # Use least squares to find the best fitting U and h
        # result = least_squares(equations, initial_guess)
        # print("radiosities:", result.x)
        # j1, j2 = result.x
        #
        # Q1 = (j1 - j2) / R12
        # print("Heat flow at surf 1:", Q1)
        # return Q1

    def threeway_rad_solver(self, T1, T2, T3, e1, e2, e3, f12, f13, f23, A1, A2, A3):
        from scipy.optimize import least_squares
        sb = 5.670374419e-8

        Eb1 = sb * T1 ** 4
        Eb2 = sb * T2 ** 4
        Eb3 = sb * T3 ** 4

        R1 = (1 - e1) / (A1 * e1)
        R2 = (1 - e2) / (A2 * e2)
        R3 = (1 - e3) / (A3 * e3)
        R12 = 1 / (A1 * f12)
        R13 = 1 / (A1 * f13)
        R23 = 1 / (A2 * f23)

        def equations(vars):
            J1, J2, J3 = vars

            eqs = [
                (Eb1 - J1) / R1 + (J2 - J1) / R12 + (J3 - J1) / R13,
                (Eb2 - J2) / R2 + (J1 - J2) / R12 + (J3 - J2) / R23,
                (Eb3 - J3) / R3 + (J1 - J3) / R13 + (J2 - J3) / R23,
            ]
            return eqs

        # Initial guess for J1, J2, J3
        initial_guess = [1, 1, 1]

        # Use least squares to find the best fitting U and h
        result = least_squares(equations, initial_guess)
        print("radiosities:", result.x)
        j1, j2, j3 = result.x

        Q1 = (j1 - j2) / R12 + (j1 - j3) / R13
        print("Heat flow at surf 1:", Q1)
        return Q1

    def box2_rad_solver(self, Tbotc, Totherc, botmat, othermat):
        assert botmat in ('foil', 'black')
        assert othermat in ('foil', 'black')

        e1 = 0.95 if botmat == 'black' else 0.05
        e2 = 0.95 if othermat == 'black' else 0.05

        Tbotk = Tbotc + 273.15
        Totherk = Totherc + 273.15

        return self.twoway_rad_solver(
            T1=Tbotk, T2=Totherk,
            e1=e1, e2=e2,
            f12=1.0,
            A1=0.25*0.25, A2=0.25*0.25*5,
        )

    def box3_rad_solver(self, Tbotc, Ttopc, botmat, midmat, topmat):
        assert botmat in ('foil', 'black')
        assert midmat in ('foil', 'black')
        assert topmat in ('foil', 'black')

        e1 = 0.95 if botmat == 'black' else 0.05
        e2 = 0.95 if midmat == 'black' else 0.05
        e3 = 0.95 if topmat == 'black' else 0.05

        Tbotk = Tbotc + 273.15
        Ttopk = Ttopc + 273.15

        Tmidk = (Tbotk + Ttopk) / 2

        return self.threeway_rad_solver(
            T1=Tbotk, T2=Tmidk, T3=Ttopk,
            e1=e1, e2=e2, e3=e3,
            f12=0.8, f13=0.2, f23=0.2,
            A1=0.25*0.25, A2=0.25*0.25*4, A3=0.25*0.25,
        )

    def fitit(self):
        from scipy.optimize import least_squares
        import numpy as np

        eq_data = [
            # 2way rad calcs
            # (10.145, 40.89, 24.65, 2.036),
            # (10.735, 42.53, 25.63, 2.147),
            # (11.353, 44.62, 26.92, 2.298),
            # (11.988, 46.7, 28.18, 2.429),
            # (12.64, 48.83, 29.37, 2.572),
            # (13.997, 53.19, 31.95, 2.887),
            # (10.179, 42.01, 23.01, 1.980),
            # (12.008, 47.47, 27.43, 2.405),
            # (10.208, 46.56, 19.13, 1.830),

            # 3way rad calcs
            (10.145, 40.89, 24.65, 1.546),
            (10.735, 42.53, 25.63, 1.627),
            (11.353, 44.62, 26.92, 1.750),
            (11.988, 46.7, 28.18, 1.851),
            (12.64, 48.83, 29.37, 1.967),
            (13.997, 53.19, 31.95, 2.207),
            (10.179, 42.01, 23.01, 1.475),
            (12.008, 47.47, 27.43, 1.826),
            (10.208, 46.56, 19.13, 1.293),

            # allfoil
            (11.675, 54.1, 27.2, 0.360),
            (11.136, 52.3, 25.7, 0.360),
            (11.307, 52.39, 25.85, 0.360),
            (10.630, 53.22, 21.23, 0.300),
        ]

        # Define the equations in a function
        def equations(vars):
            U, h, f1, f2 = vars

            eqs = []
            for real_value, room_bot_t, bot_top_t, rad_est in eq_data:
                rad = rad_est

                eq = U * 0.0625 * room_bot_t + h * 0.0625 * bot_top_t + rad - real_value
                eqs.append(eq)

            eqs.append(h - 2.37)
            eqs.append(U - 1.84)
            return eqs

        # Initial guess for U and h
        initial_guess = [1, 1, 0, 0]

        # Use least squares to find the best fitting U and h
        result = least_squares(equations, initial_guess)
        print(result.x)


if __name__ == '__main__':
    fire.Fire(CmdLine)
