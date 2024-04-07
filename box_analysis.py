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


if __name__ == '__main__':
    fire.Fire(CmdLine)
