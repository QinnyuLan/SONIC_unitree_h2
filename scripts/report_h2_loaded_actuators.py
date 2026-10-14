#!/usr/bin/env python3
"""Static scientific figures for conditional loaded actuator experiments."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from identify_h2_loaded_actuators import Replay, GROUPS

COLORS = ['#2276a5', '#16806a', '#c34b3b', '#875494']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--candidate', type=Path)
    p.add_argument('--nuisance', default='nuisance01')
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    screen = json.loads((a.root/'screen03/screen.json').read_text())
    fit = json.loads((a.candidate or a.root/'fit01/candidate.json').read_text())
    rows = list(csv.DictReader((a.root/a.nuisance/'windows.csv').open()))
    nominal = next(r for r in screen if r['name'] == 'nominal')
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout='constrained')
    for group, color in zip(GROUPS, COLORS):
        for ax, key, default, xlabel in zip(axes, ['gain', 'time_constant_ms'], [1, 0],
                    ['Motor effort gain (ratio)', 'First-order time constant (ms)']):
            values = [(default, nominal['validation_score'])]
            values += [(r['config'][group][key], r['validation_score']) for r in screen
                       if group in r['config'] and key in r['config'][group]]
            values.sort()
            x, y = np.array(values).T
            ax.plot(x, y, 'o-', color=color, label=group.replace('_', ' '), markersize=4)
            ax.set(xlabel=xlabel, ylabel='Held-out composite mean-square score', ylim=(0, None))
            ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Single-group sensitivity | repetition 3, 250 ms windows\nOther motor parameters and requested PD gains unchanged')
    fig.savefig(a.output/'motor_sensitivity.png', dpi=160)
    plt.close(fig)

    nuisance = []
    for velocity, clearance in dict.fromkeys((r['velocity'], r['clearance']) for r in rows):
        vals = {name: np.mean([float(r['score']) for r in rows if r['candidate'] == name
                             and r['velocity'] == velocity and r['clearance'] == clearance])
                for name in ('nominal', 'candidate')}
        nuisance.append(dict(velocity=velocity, clearance_m=float(clearance), **vals,
                             relative_improvement=1-vals['candidate']/vals['nominal']))
    fig, ax = plt.subplots(figsize=(9, 4), layout='constrained')
    x = np.arange(len(nuisance))
    ax.bar(x-.18, [r['nominal'] for r in nuisance], .36, color=COLORS[0], label='Existing ankle model')
    ax.bar(x+.18, [r['candidate'] for r in nuisance], .36, color=COLORS[1], label='Plus left-knee candidate')
    ax.set_xticks(x, [r['velocity']+'\n'+f"{r['clearance_m']*1000:g} mm" for r in nuisance])
    ax.set(xlabel='Root-velocity assumption / minimum sole clearance',
           ylabel='Composite mean-square score', title='Sensitivity to unmeasured initial state | all 12 windows')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=.2)
    fig.savefig(a.output/'initial_state_sensitivity.png', dpi=160)
    plt.close(fig)

    replay = Replay(a.root/'loaded_dataset02/dataset.npz')
    traces = np.load(a.root/a.nuisance/'traces.npz')
    window = next(w for w in replay.windows if w['label'] == 'rep3_rise2')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), layout='constrained')
    specs = [(3, 'Left knee angle', 'Angle (deg)'), (34, 'Left knee velocity', 'Velocity (deg/s)'),
             (62, 'Pelvis pitch', 'Angle (deg)'), (5, 'Left ankle pitch', 'Angle (deg)')]
    t = window['times'][::5] * 1000
    for ax, (index, title, ylabel) in zip(axes.flat, specs):
        measured = window['pitch'] if index == 62 else window['measured'][:, index]
        ax.plot(t, np.rad2deg(measured), color=COLORS[2], label='Real measurement', linewidth=2)
        for name, label, color in [('nominal', 'Existing ankle model', COLORS[0]),
                                   ('candidate', 'Left-knee candidate', COLORS[1])]:
            ax.plot(t, np.rad2deg(traces[name+'_'+window['label']][:, index]), color=color, label=label)
        ax.set(xlabel='Time since window start (ms)', ylabel=ylabel, title=title)
        ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle('Held-out repetition 3, second rise | same recorded LowCmd\nOne initialization, no measured-state injection; not a full-body rollout')
    fig.savefig(a.output/'short_window_validation.png', dpi=160)
    plt.close(fig)

    heldout = {}
    for name in ('nominal', 'candidate'):
        selected = [r for r in rows if r['candidate'] == name and r['repeat'] == '3'
                    and r['velocity'] == 'both' and float(r['clearance']) == 0]
        keys = ['score', 'pitch_rmse_deg'] + [g+'_q_rmse_deg' for g in GROUPS]
        heldout[name] = {key: float(np.mean([float(r[key]) for r in selected])) for key in keys}
    report = dict(candidate=fit, screen_nominal=nominal, heldout_window_means=heldout,
                  nuisance=nuisance,
                  interpretation='conditional effective actuator candidate; not measured motor efficiency or delay')
    (a.output/'summary.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
