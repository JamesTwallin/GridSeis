"""Train the weather house's on-board neural network and turn it into a TensorFlow Lite Micro model.

Run with the TensorFlow environment on D: (python 3.12, tensorflow-cpu 2.17):
    D:/gridseis_hum/tfenv/Scripts/python.exe hardware/tools/tinyml_train.py search   compare network shapes
    D:/gridseis_hum/tfenv/Scripts/python.exe hardware/tools/tinyml_train.py build    train the chosen one, write firmware

Inputs come from tinyml_inputs.npz, built by tinyml_inputs.py from the band data on D: (only the board's own
spectrum: 13 bands, each placed against its own last seven days; no clock):
  A  52 numbers: each band's latest placing and its 1, 6 and 24 hour averages
  B  312 numbers: each band's hourly average placing for each of the last 24 hours (13 bands x 24 hours)
Target: where the half hour's NESO carbon intensity sits in its previous seven days (0 dirtiest, 1 cleanest).

The check: trained on July 2021 to December 2023 (the last six months held back to decide when to stop),
then run unchanged on 2024 to September 2026, as the weighted sum was."""
import os, sys, json
import numpy as np, pandas as pd
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__)); HW = os.path.dirname(HERE)
NPZ = 'D:/gridseis_hum/hardware_test/tinyml_inputs.npz'
NB, HOURS = 13, 24
tf.keras.utils.set_random_seed(0)


def load():
    z = np.load(NPZ, allow_pickle=True)
    return z['A'].astype(np.float32), z['B'].astype(np.float32), z['y'].astype(np.float32), z['ci'], pd.DatetimeIndex(z['t'])


def score(p, y, t):
    out = {}
    for name, (a, b) in {'2024': ('2024', '2025'), '2025': ('2025', '2026'), '2026': ('2026', '2026-10'),
                         'all': ('2024', '2026-10')}.items():
        m = (t >= pd.Timestamp(a)) & (t < pd.Timestamp(b))
        out[name] = round(float(spearmanr(p[m], y[m]).statistic), 3)
    return out


def dense(n_in, hidden):
    inp = tf.keras.Input((n_in,)); x = inp
    for h in hidden: x = tf.keras.layers.Dense(h, activation='relu')(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(x))


def conv(ch=16):
    # 13 bands x 24 hours, oldest hour last in the input; reshape to (hours, bands) for 1D convolution over time
    inp = tf.keras.Input((NB * HOURS,))
    x = tf.keras.layers.Reshape((NB, HOURS))(inp); x = tf.keras.layers.Permute((2, 1))(x)
    x = tf.keras.layers.Conv1D(ch, 3, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(ch, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(ch, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x); x = tf.keras.layers.Dense(16, activation='relu')(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1)(x))


def fit(model, X, y, tr, va):
    model.compile(tf.keras.optimizers.AdamW(2e-3, weight_decay=1e-3), 'mse')
    model.fit(X[tr], y[tr], validation_data=(X[va], y[va]), epochs=80, batch_size=256, verbose=0,
              callbacks=[tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)])
    return model


def split(t, end):
    tr = t < pd.Timestamp(end) - pd.DateOffset(months=6); va = (t >= pd.Timestamp(end) - pd.DateOffset(months=6)) & (t < pd.Timestamp(end))
    return tr, va


def candidates(A, B):
    return {'A: 52 inputs, 32-16 network': (A, lambda: dense(A.shape[1], [32, 16])),
            'A: 52 inputs, 64-32 network': (A, lambda: dense(A.shape[1], [64, 32])),
            'B: 312 inputs, 32-16 network': (B, lambda: dense(B.shape[1], [32, 16])),
            'B: 312 inputs, 1D convolution over the 24 hours': (B, lambda: conv())}


def search():
    A, B, y, ci, t = load(); te = t >= pd.Timestamp('2024-01-01'); tr, va = split(t, '2024-01-01')
    res = {}
    lin = [k for k in range(A.shape[1]) if k % 4 != 0]
    w = np.linalg.lstsq(np.c_[A[tr | va][:, lin], np.ones((tr | va).sum())], y[tr | va], rcond=None)[0]
    res['weighted sum (current firmware)'] = score(np.c_[A[te][:, lin], np.ones(te.sum())] @ w, y[te], t[te])
    print('weighted sum', res['weighted sum (current firmware)'], flush=True)
    for name, (X, make) in candidates(A, B).items():
        runs = []
        for seed in range(3):   # three starts, to see how much the result moves by chance
            tf.keras.utils.set_random_seed(seed); m = fit(make(), X, y, tr, va)
            runs.append(score(m.predict(X[te], verbose=0)[:, 0], y[te], t[te]))
        res[name] = {k: round(float(np.mean([r[k] for r in runs])), 3) for k in runs[0]}
        res[name]['spread_between_starts_all'] = round(float(np.ptp([r['all'] for r in runs])), 3)
        res[name]['parameters'] = int(m.count_params())
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(HERE, 'tinyml_search_results.json'), 'w'), indent=1)


def board_net(ch=16):
    """The deployed network. Input is (24 hours, 13 bands) as the board lays it out: row 0 is the last hour."""
    inp = tf.keras.Input((HOURS, NB), name='hours_by_band')
    x = tf.keras.layers.Conv1D(ch, 3, padding='same', activation='relu')(inp)
    x = tf.keras.layers.Conv1D(ch, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Conv1D(ch, 3, strides=2, padding='same', activation='relu')(x)
    x = tf.keras.layers.Flatten()(x); x = tf.keras.layers.Dense(16, activation='relu')(x)
    return tf.keras.Model(inp, tf.keras.layers.Dense(1, name='cleanliness')(x))


def to_board_layout(B):
    return B.reshape(-1, NB, HOURS).transpose(0, 2, 1).copy()   # npz columns are band-major, hour 0 first


def int8_model(model, X_rep):
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    conv.optimizations = [tf.lite.Optimize.DEFAULT]
    rng = np.random.default_rng(0); idx = rng.choice(len(X_rep), 2000, replace=False)
    conv.representative_dataset = lambda: ([X_rep[i:i + 1]] for i in idx)
    conv.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    conv.inference_input_type = tf.int8; conv.inference_output_type = tf.int8
    return conv.convert()


def run_tflite(blob, X):
    it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
    i, o = it.get_input_details()[0], it.get_output_details()[0]
    si, zi = i['quantization']; so, zo = o['quantization']; out = np.empty(len(X), np.float32)
    for k in range(len(X)):
        it.set_tensor(i['index'], np.clip(np.round(X[k:k + 1] / si + zi), -128, 127).astype(np.int8)); it.invoke()
        out[k] = (it.get_tensor(o['index'])[0, 0].astype(np.float32) - zo) * so
    return out


def build():
    A, B, y, ci, t = load(); X = to_board_layout(B); te = t >= pd.Timestamp('2024-01-01'); res = {}
    # 1. frozen check of the deployed shape, as float and as int8
    tr, va = split(t, '2024-01-01'); tf.keras.utils.set_random_seed(0); m = fit(board_net(), X, y, tr, va)
    blob = int8_model(m, X[tr])
    res['check_float'] = score(m.predict(X[te], verbose=0)[:, 0], y[te], t[te])
    p8 = run_tflite(blob, X[te]); res['check_int8'] = score(p8, y[te], t[te])
    lo, mid, hi = np.percentile(run_tflite(blob, X[tr | va]), [10, 50, 90])
    fifth = pd.qcut(p8, 5, labels=False); tt = pd.Series(y[te])
    res['check_int8_station_out_dirtier'] = round(float((tt[fifth == 0] < 0.5).mean()), 3)
    res['check_int8_turbine_out_cleaner'] = round(float((tt[fifth == 4] > 0.5).mean()), 3)
    res['check_int8_gco2_by_fifth'] = pd.Series(ci[te]).groupby(fifth).mean().round(0).tolist()
    print('frozen check', res, flush=True)
    # 2. final model: everything up to March 2026 to learn, April to September 2026 to decide when to stop
    tr, va = split(t, '2026-10-01'); tf.keras.utils.set_random_seed(0); m = fit(board_net(), X, y, tr, va)
    blob = int8_model(m, X[tr]); p = run_tflite(blob, X)
    lo, mid, hi = np.percentile(p[tr | va], [10, 50, 90])
    res.update(final_bytes=len(blob), final_parameters=int(m.count_params()), position_scale=[float(lo), float(mid), float(hi)])
    it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
    qi, qo = it.get_input_details()[0]['quantization'], it.get_output_details()[0]['quantization']
    ops = sorted({d['op_name'] for d in it._get_ops_details()}); res['ops'] = ops
    # three inputs from the test years with their expected outputs, for the board's start-up self-test
    pick = np.where(te)[0][[1000, 20000, 40000]]
    write_header(blob, m, X, p, pick, lo, mid, hi, qi, qo, ops, 'steady state only: needs 24 hours of placings')
    json.dump(res, open(os.path.join(HERE, 'tinyml_results.json'), 'w'), indent=1)
    print('final', {k: res[k] for k in ('final_bytes', 'final_parameters', 'position_scale', 'ops')})


def write_header(blob, m, X, p, pick, lo, mid, hi, qi, qo, ops, note):
    with open(os.path.join(HW, 'firmware', 'src', 'nn_model_data.h'), 'w') as f:
        f.write('// Generated by hardware/tools/tinyml_train.py - do not edit by hand.\n')
        f.write('// int8 TensorFlow Lite model of the weather house network: 24 hours x 13 bands in, cleanliness out.\n')
        f.write(f'// Trained on NESO frequency (2.5 mHz noise added) and carbon intensity, July 2021 to September 2026.\n')
        f.write(f'// {note}.\n// {m.count_params()} weights. Ops: {", ".join(ops)}\n#pragma once\n#include <stdint.h>\n\n')
        f.write(f'#define NN_HOURS {HOURS}\n#define NN_BANDS {NB}\n')
        f.write(f'#define NN_IN_SCALE {qi[0]:.9g}f\n#define NN_IN_ZERO {qi[1]}\n#define NN_OUT_SCALE {qo[0]:.9g}f\n#define NN_OUT_ZERO {qo[1]}\n')
        f.write(f'#define NN_P_LO {lo:.6f}f\n#define NN_P_MID {mid:.6f}f\n#define NN_P_HI {hi:.6f}f\n\n')
        f.write(f'alignas(16) static const unsigned char g_nn_model[{len(blob)}] = {{\n')
        for k in range(0, len(blob), 16): f.write('    ' + ', '.join(f'0x{b:02x}' for b in blob[k:k + 16]) + ',\n')
        f.write('};\n\n#define NN_SELFTEST_N 3\n')
        f.write('static const float g_nn_selftest_in[NN_SELFTEST_N][NN_HOURS * NN_BANDS] = {\n')
        for i in pick: f.write('    {' + ', '.join(f'{v:.6f}f' for v in X[i].ravel()) + '},\n')
        f.write('};\nstatic const float g_nn_selftest_out[NN_SELFTEST_N] = {' + ', '.join(f'{p[i]:.6f}f' for i in pick) + '};\n')


# ---------------------------------------------------------------- the network that also works in its first hours
WARM_NPZ = 'D:/gridseis_hum/hardware_test/tinyml_warm_inputs.npz'
AGES = {'3 to 6 h': (0, 6), '6 to 12 h': (6, 12), '12 to 24 h': (12, 24), '1 to 2 days': (24, 48),
        '2 to 7 days': (48, 168), 'over 7 days': (168, 1e9)}


def load_warm():
    z = np.load(WARM_NPZ, allow_pickle=True)
    return (z['X'].astype(np.float32), z['y'].astype(np.float32), z['ci'], pd.DatetimeIndex(z['t']),
            z['elapsed_h'].astype(np.float32))


def score_by_age(p, y, el):
    out = {}
    for name, (a, b) in AGES.items():
        m = (el >= a) & (el < b)
        if m.sum() > 200: out[name] = round(float(spearmanr(p[m], y[m]).statistic), 3)
    return out


def build_warm():
    """Train on inputs as a board sees them from 3 hours after switch-on (tinyml_inputs.py warm), check
    frozen on 2024 to 2026 by hours since switch-on, then train on everything and write the firmware."""
    X, y, ci, t, el = load_warm(); te = t >= pd.Timestamp('2024-01-01'); res = {}
    tr, va = split(t, '2024-01-01'); tf.keras.utils.set_random_seed(0); m = fit(board_net(), X, y, tr, va)
    blob = int8_model(m, X[tr]); p8 = run_tflite(blob, X[te])
    res['check_int8_by_hours_since_switch_on'] = score_by_age(p8, y[te], el[te])
    res['check_int8_by_year_all_ages'] = score(p8, y[te], t[te])
    steady = el[te] >= 168
    res['check_int8_by_year_steady'] = score(p8[steady], y[te][steady], t[te][steady])
    print('frozen check', res, flush=True)
    tr, va = split(t, '2026-10-01'); tf.keras.utils.set_random_seed(0); m = fit(board_net(), X, y, tr, va)
    blob = int8_model(m, X[tr]); p = run_tflite(blob, X)
    lo, mid, hi = np.percentile(p[(tr | va) & (el >= 24)], [10, 50, 90])
    res.update(final_bytes=len(blob), final_parameters=int(m.count_params()), position_scale=[float(lo), float(mid), float(hi)])
    it = tf.lite.Interpreter(model_content=blob); it.allocate_tensors()
    qi, qo = it.get_input_details()[0]['quantization'], it.get_output_details()[0]['quantization']
    ops = sorted({d['op_name'] for d in it._get_ops_details()}); res['ops'] = ops
    idx = np.where(te)[0]; pick = [idx[1000], idx[20000], idx[np.argmax(el[te] < 6)]]   # one from the first hours
    write_header(blob, m, X, p, pick, lo, mid, hi, qi, qo, ops,
                 'Trained on stretches that start from an empty history, so it works from 3 hours after switch-on')
    json.dump(res, open(os.path.join(HERE, 'tinyml_warm_results.json'), 'w'), indent=1)
    print('final', {k: res[k] for k in ('final_bytes', 'final_parameters', 'position_scale', 'ops')})
    # figure: match against hours since switch-on
    sys.path.insert(0, os.path.join(HW, '..', 'scripts', 'primer'))
    from style import tidy, title, INK
    import matplotlib.pyplot as plt
    ages = res['check_int8_by_hours_since_switch_on']
    fig, ax = plt.subplots(figsize=(10, 4.4))
    ax.bar(range(len(ages)), list(ages.values()), color='#0f766e', width=0.6)
    for i, v in enumerate(ages.values()): ax.text(i, v + 0.01, f'{v:.2f}', ha='center', fontsize=9, color=INK)
    ax.set_xticks(range(len(ages))); ax.set_xticklabels(list(ages.keys())); ax.set_ylim(0, 1); tidy(ax)
    ax.set_xlabel('time since the board was switched on'); ax.set_ylabel('how closely it follows the real week\n(1 would be a perfect match)')
    title(fig, 'On-board network: match with real cleanliness by time since switch-on, 2024 to 2026', y=0.99, size=14)
    fig.text(0.01, -0.03, 'Trained on July 2021 to December 2023 and run unchanged on 2024 to September 2026, on stretches that each '
             'start from an empty history. Data: NESO system frequency (1 s) with 2.5 mHz of noise added; NESO carbon intensity.',
             fontsize=8, color='grey')
    fig.tight_layout(rect=(0, 0, 1, 0.93)); fig.savefig(os.path.join(HW, 'images', 'tinyml_warmup.png'), dpi=200, bbox_inches='tight', pad_inches=0.25)
    print('saved tinyml_warmup.png')


def plot():
    """hardware/images/tinyml_check.png: the int8 network from the frozen check against real cleanliness,
    with the older one-band rule and the weighted sum for comparison."""
    sys.path.insert(0, os.path.join(HW, '..', 'scripts', 'primer'))
    from style import tidy, title, INK, GREY
    import matplotlib.pyplot as plt, matplotlib.dates as mdates
    TEAL, ORANGE, BLUE = '#0f766e', '#c2410c', '#1d4ed8'
    A, B, y, ci, t = load(); X = to_board_layout(B); te = t >= pd.Timestamp('2024-01-01')
    tr, va = split(t, '2024-01-01'); tf.keras.utils.set_random_seed(0); m = fit(board_net(), X, y, tr, va)
    blob = int8_model(m, X[tr]); lo, mid, hi = np.percentile(run_tflite(blob, X[tr | va]), [10, 50, 90])
    nn8 = run_tflite(blob, X[te])
    lin = [k for k in range(A.shape[1]) if k % 4 != 0]
    w = np.linalg.lstsq(np.c_[A[tr | va][:, lin], np.ones((tr | va).sum())], y[tr | va], rcond=None)[0]
    ws = np.c_[A[te][:, lin], np.ones(te.sum())] @ w
    rule = pd.read_parquet(f'{os.path.dirname(NPZ)}/model_features_thin.parquet', columns=['rule']).rule
    rule = rule.reindex(t[te]).values
    d = pd.DataFrame({'real': y[te], 'ci': ci[te], 'rule': rule, 'sum': ws, 'nn': nn8}, index=t[te]).dropna()
    pos = np.clip(np.where(d.nn < mid, (d.nn - mid) / (mid - lo), (d.nn - mid) / (hi - mid)), -1, 1)
    d['nn_shown'] = (pos + 1) / 2
    fig = plt.figure(figsize=(13, 9.6)); g = fig.add_gridspec(2, 2, height_ratios=[1, 1.05], hspace=0.42, wspace=0.22)
    a = fig.add_subplot(g[0, :]); x = d['2025-04-21':'2025-05-04 23:30']
    a.plot(x.index, x.real, color=ORANGE, lw=1.8, label='real: carbon intensity against its previous 7 days')
    a.plot(x.index, x.nn_shown, color=TEAL, lw=1.8, label='on-board network (int8), as the figures show it')
    a.set_ylim(0, 1); a.set_yticks([0, 0.5, 1]); a.set_yticklabels(['dirtiest /\npower station out', 'middle /\nboth inside', 'cleanest /\nturbine out'])
    a.xaxis.set_major_formatter(mdates.DateFormatter('%a %d %b')); tidy(a)
    a.legend(frameon=False, loc='upper left', ncols=2, fontsize=9.5, bbox_to_anchor=(0, 1.14))
    a.set_title('21 April to 4 May 2025, half-hourly', loc='left', fontsize=12, color=INK, pad=28)
    methods = [('rule', GREY, 'older rule, one band'), ('sum', BLUE, 'weighted sum, 13 bands'), ('nn', TEAL, 'on-board network, 13 bands')]
    b = fig.add_subplot(g[1, 0]); years = ['2024', '2025', '2026']
    for k, (col, colour, label) in enumerate(methods):
        v = [spearmanr(d.loc[yr, col], d.loc[yr, 'real']).statistic for yr in years]
        b.bar(np.arange(3) + (k - 1) * 0.27, v, 0.25, color=colour, label=label)
        for xi, vi in enumerate(v): b.text(xi + (k - 1) * 0.27, vi + 0.01, f'{vi:.2f}', ha='center', fontsize=8.5, color=INK)
    b.set_xticks(range(3)); b.set_xticklabels(['2024', '2025', '2026 (to Sep)']); b.set_ylim(0, 1); tidy(b)
    b.set_ylabel('how closely it follows the real week\n(1 would be a perfect match)'); b.legend(frameon=False, loc='upper right', fontsize=9)
    b.set_title('Match with real cleanliness, by year', loc='left', fontsize=12, color=INK)
    c = fig.add_subplot(g[1, 1])
    for k, (col, colour, label) in enumerate(methods):
        gg = d.ci.groupby(pd.qcut(d[col], 5, labels=False)).mean()
        c.bar(np.arange(5) + (k - 1) * 0.27, gg.values, 0.25, color=colour, label=label)
        for xi, vi in enumerate(gg.values): c.text(xi + (k - 1) * 0.27, vi + 2, f'{vi:.0f}', ha='center', fontsize=7.5, color=INK)
    c.set_xticks(range(5)); c.set_xticklabels(['power station\nfully out', 'half out', 'both inside', 'half out', 'turbine\nfully out'], fontsize=9)
    c.set_ylabel('average NESO carbon intensity, gCO$_2$/kWh'); tidy(c)
    c.set_title('Carbon intensity at each figure position, 2024 to Sep 2026', loc='left', fontsize=12, color=INK)
    title(fig, 'On-board network against real cleanliness, 2024 to 2026', y=0.985, size=16)
    fig.text(0.01, 0.005, 'All three were fitted on July 2021 to December 2023 and run unchanged on 2024 to September 2026. '
             'Inputs: only the spectrum of NESO one-second frequency with 2.5 mHz of noise added.\nEach figure position is a fifth of '
             'the half-hours. Data: NESO system frequency; NESO carbon intensity.', fontsize=8, color='grey')
    fig.savefig(os.path.join(HW, 'images', 'tinyml_check.png'), dpi=200, bbox_inches='tight', pad_inches=0.25)
    print('saved tinyml_check.png')


if __name__ == '__main__':
    {'search': search, 'build': build, 'plot': plot, 'warm': build_warm}[sys.argv[1]]()
