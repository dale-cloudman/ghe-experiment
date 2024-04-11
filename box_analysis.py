import pprint

import numpy as np
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

            mean = df[(df['Date/Time'] >= start_time) & (df['Date/Time'] < end_time)][col].mean()
            std = df[(df['Date/Time'] >= start_time) & (df['Date/Time'] < end_time)][col].std()
            # use %.3f for power, otherwise %.1f
            if col == 'HPPower':
                print("%-15s: %6.3f - %6.3f, %6.3f +/- %6.3f" % (col, minval, maxval, mean, 2*std))
            else:
                print("%-15s: %6.1f - %6.1f, %6.2f +/- %6.2f" % (col, minval, maxval, mean, 2*std))

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
        # print("radiosities:", result.x)
        j1, j2, j3 = result.x

        Q1 = (j1 - j2) / R12 + (j1 - j3) / R13
        # print("Heat flow at surf 1:", Q1)
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

    def fit_topc_given_botc_and_power(self):
        # given data with powers and bottom temp and room temp
        # predict top temp
        pass

    def pred(self, power_w, Troomc):
        U = 2.151
        h = 2.245

        Troomk = Troomc + 273.15

        for room_botk in np.arange(30, 70, 1):
            Tbotk = Troomk + room_botk

            errs = []
            for bot_topk in np.arange(0, 50, 0.1):
                Ttopk = Tbotk - bot_topk

                cond = U * 0.0625 * (Tbotk - Troomk)
                conv = h * 0.0625 * (Tbotk - Ttopk)
                rad = self.box3_rad_solver(
                    Tbotc=Tbotk - 273.15,
                    Ttopc=Ttopk - 273.15,
                    botmat='black',
                    midmat='foil',
                    topmat='foil',
                )

                pred_w = cond + conv + rad
                # print("Tbotk - Troomk:", Tbotk - Troomk)
                # print("Tbotk - Ttopk:", Tbotk - Ttopk)
                # print("cond + conv + rad = %.2f + %.2f + %.2f = %.2f" % (cond, conv, rad, pred_w))
                # print("-", Tbotc, Ttopc, pred_w)
                err = abs(power_w - pred_w)
                errs.append({
                    'room_botk': room_botk,
                    'bot_topk': bot_topk,
                    'err': err,
                })

            best = min(errs, key=lambda x: abs(x['err']))
            print("Best for room<->bot=%.1fK: bot<->top=%.1fK, bot=%.1fc, top=%.1fc @ err=%.3f" % (
                best['room_botk'],
                best['bot_topk'],
                Troomk + best['room_botk'] - 273.15,
                Troomk + best['room_botk'] - best['bot_topk'] - 273.15,
                best['err'],
            ))

    def pred2(self, current, roomC, topC):
        U = 2.151
        h = 2.245

        resf = open("tmp.csv", "w")

        resf.write("roomC,topC,predbotC\n")

        for roomC in [23.5, 24, 24.5, 25, 25.5]:
            for topC in np.arange(54, 74, 0.1):
                errs = []
                for bottempC in np.arange(70, 95, 0.1):
                    resistance = 0.0031777907*bottempC + 2.9267924598
                    power = current * current * resistance

                    cond = U * 0.0625 * (bottempC - roomC)
                    conv = h * 0.0625 * (bottempC - topC)
                    rad = self.box3_rad_solver(
                        Tbotc=bottempC,
                        Ttopc=topC,
                        botmat='black',
                        midmat='foil',
                        topmat='foil',
                    )

                    pred_w = cond + conv + rad
                    err = power - pred_w
                    errs.append({
                        'bottempC': bottempC,
                        'err': err,
                        'power': power,
                        'pred_w': pred_w,
                    })
                    # print("For bot=%.1fC, predicted power=%.3fW, predicted heat loss=%.3fW, err=%.3f" % (
                    #     bottempC, power, pred_w, err
                    # ))

                best = min(errs, key=lambda x: abs(x['err']))
                print("If room=%.1fC and current=%.3fA and top=%.1fC, then bottom predicted to be %.1fC" % (
                    roomC, current, topC, best['bottempC']
                ))
                resf.write(f"{roomC:.2f},{topC:.2f},{best['bottempC']:.2f}\n")

    def platesolver_v1(self):
        from scipy.optimize import least_squares

        sky = 240
        solar1 = 1359 - 1250
        solar2 = 1250 - 1150
        solar3 = 1150 - 1058
        solar4 = 1058 - 974
        solarB = 974

        def equations(vars):
            P1, P2, P3, P4, PB = vars

            eqs = [
                # top glass in = out
                solar1 + P2 + sky - 2*P1,
                solar2 + P3 + P1 - 2*P2,
                solar3 + P4 + P2 - 2*P3,
                solar4 + PB + P3 - 2*P4,
                # bottom plate
                solarB + P4 - PB,
            ]
            return eqs

        initial_guess = [100, 100, 100, 100, 100]

        # Use least squares to find the best fitting U and h
        result = least_squares(equations, initial_guess)
        print("resuts:", result.x)

    def platesolver_v2(self):
        from scipy.optimize import least_squares

        sky = 240
        solar1 = 0
        solar2 = 0
        solar3 = 0
        solar4 = 0
        solarB = 974

        def equations(vars):
            P1, P2, P3, P4, PB = vars

            eqs = [
                # top glass in = out
                solar1 + P2 + sky - 2*P1,
                solar2 + P3 + P1 - 2*P2,
                solar3 + P4 + P2 - 2*P3,
                solar4 + PB + P3 - 2*P4,
                # bottom plate
                solarB + P4 - PB,
            ]
            return eqs

        initial_guess = [100, 100, 100, 100, 100]

        # Use least squares to find the best fitting U and h
        result = least_squares(equations, initial_guess)
        print("resuts:", result.x)

    def platesolver_v3(self):
        from scipy.optimize import least_squares

        for f in np.arange(0.01, 1.00, 0.01):
            sky = 240
            # solar1 = 1359 - 1250
            # solar2 = 1250 - 1150
            # solar3 = 1150 - 1058
            # solar4 = 1058 - 974
            solar1 = 0
            solar2 = 0
            solar3 = 0
            solar4 = 0
            solarB = 974
            bf = 1.0
            def equations(vars):
                P1, P2, P3, P4, PB = vars

                eqs = [
                    solar1 + sky*f + P2*f + P3*(1-f)*f + P4*(1-f)**2*f + PB*(1-f)**3*f - 2*P1,
                    solar2 + sky*(1-f)*f + P1*f + P3*f + P4*(1-f)*f + PB*(1-f)**2*f - 2*P2,
                    solar3 + sky*(1-f)**2*f + P1*(1-f)*f + P2*f + P4*f + PB*(1-f)*f - 2*P3,
                    solar4 + sky*(1-f)**3*f + P1*(1-f)**2*f + P2*(1-f)*f + P3*f + PB*f - 2*P4,
                    solarB + sky*(1-f)**4*bf + P1*(1-f)**3*bf + P2*(1-f)**2*bf + P3*(1-f)*bf + P4*bf - PB,
                ]
                return eqs

            initial_guess = [100, 100, 100, 100, 100]

            # Use least squares to find the best fitting U and h
            result = least_squares(equations, initial_guess)

            sb = 5.670374419e-8
            fs = [f, f, f, f, bf]
            temps = np.array([(x / (sb * thisf))**(1/4)-273.15 for x, thisf in zip(result.x, fs)])

            print("For f=%.2f, results=%s, temps=%s" % (
                f, result.x, temps
            ))

    def platesolver_condconv(self):
        # assuming 100% IR absortivity
        from scipy.optimize import least_squares

        # inputs
        solar1 = 1359 - 1250
        solar2 = 1250 - 1150
        solar3 = 1150 - 1058
        solar4 = 1058 - 974
        solarB = 974

        sky = 240

        TairC = 2  # very cold!

        # conductance bottom to outside
        U = 5.1
        # inner convection
        h_inner = 2.5
        # convection with outside air
        h_outer = 12
        # absorption fraction of glass of ir, rest is transmitted
        f_ir = 0.92
        bf_ir = 1.00  # black plate abosrb 100%

        # constants
        sb = 5.670374419e-8

        def equations(vars):
            T1, T2, T3, T4, TB = vars

            P1rad = sb * f_ir * (T1 + 273.15) ** 4
            P1conv = h_outer * (T1 - TairC)

            P2rad = sb * f_ir * (T2 + 273.15) ** 4
            P2conv = h_inner * (T2 - T1)

            P3rad = sb * f_ir * (T3 + 273.15) ** 4
            P3conv = h_inner * (T3 - T2)

            P4rad = sb * f_ir * (T4 + 273.15) ** 4
            P4conv = h_inner * (T4 - T3)

            PBrad = sb * bf_ir * (TB + 273.15) ** 4
            PBconv = h_inner * (TB - T4)
            PBcond = U * (TB - TairC)

            print("P1rad=%.2f, P1conv=%.2f" % (P1rad, P1conv))
            print("P2rad=%.2f, P2conv=%.2f" % (P2rad, P2conv))
            print("P3rad=%.2f, P3conv=%.2f" % (P3rad, P3conv))
            print("P4rad=%.2f, P4conv=%.2f" % (P4rad, P4conv))
            print("PBrad=%.2f, PBconv=%.2f, PBcond=%.2f" % (PBrad, PBconv, PBcond))

            assert bf_ir == 1.00
            eqs = [
                # P1 gains its portion of solar, its portion of the radiation from sky and all else, and convection from P2
                # P1 loses its convection to the outside air, and its own emission
                (solar1 + P2conv + sky*f_ir + P2rad*f_ir+ P3rad*(1-f_ir)*f_ir + P4rad*(1-f_ir)**2*f_ir + PBrad*(1-f_ir)**3*f_ir) - (P1conv + 2*P1rad),
                # P2-P4 similarly
                (solar2 + P3conv + sky*(1-f_ir)*f_ir + P1rad*f_ir + P3rad*f_ir + P4rad*(1-f_ir)*f_ir + PBrad*(1-f_ir)**2*f_ir) - (P2conv + 2*P2rad),
                (solar3 + P4conv + sky*(1-f_ir)**2*f_ir + P1rad*(1-f_ir)*f_ir + P2rad*f_ir + P4rad*f_ir + PBrad*(1-f_ir)*f_ir) - (P3conv + 2*P3rad),
                (solar4 + PBconv + sky*(1-f_ir)**3*f_ir + P1rad*(1-f_ir)**2*f_ir + P2rad*(1-f_ir)*f_ir + P3rad*f_ir + PBrad*f_ir) - (P4conv + 2*P4rad),
                # bottom: gains from solar & P4rad, loses from cond and conv and rad
                (solarB + sky*(1-f_ir)**4*bf_ir + P1rad*(1-f_ir)**3*bf_ir + P2rad*(1-f_ir)**2*bf_ir + P3rad*(1-f_ir)*bf_ir + P4rad*bf_ir) - (PBcond + PBconv + PBrad),
            ]
            return eqs

        initial_guess = [10]*5

        # Use least squares to find the best fitting U and h
        result = least_squares(equations, initial_guess)

        temps = result.x
        fs = [f_ir, f_ir, f_ir, f_ir, bf_ir]
        emissions = np.array([sb * f * (x + 273.15) ** 4 for x, f in zip(temps, fs)])

        print("temps=%s, emissions=%s" % (
            temps, emissions,
        ))

        T1, T2, T3, T4, TB = temps
        P1conv = h_outer * (T1 - TairC)
        P2conv = h_inner * (T2 - T1)
        P3conv = h_inner * (T3 - T2)
        P4conv = h_inner * (T4 - T3)
        PBconv = h_inner * (TB - T4)

        PBcond = U * (TB - TairC)

        print("CONv losses: P1=%.2f, P2=%.2f, P3=%.2f, P4=%.2f, PB=%.2f" % (
            P1conv, P2conv, P3conv, P4conv, PBconv
        ))
        print("Cond losses from bottom to outside: %.2f" % PBcond)

        T_between_12 = (T1 + T2) / 2
        T_between_23 = (T2 + T3) / 2
        T_between_34 = (T3 + T4) / 2
        T_between_4B = (T4 + TB) / 2
        print("After glass 1:", T_between_12)
        print("After glass 2:", T_between_23)
        print("After glass 3:", T_between_34)
        print("After glass 4:", T_between_4B)


if __name__ == '__main__':
    fire.Fire(CmdLine)
