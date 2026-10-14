#!/usr/bin/env python3
"""Profile a conditional loaded-knee candidate without selecting on held-out data.

All work is offline MuJoCo. Scores are descriptive within-session replay losses,
not confidence intervals or calibrated likelihoods for hardware parameters.
"""

import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from identify_h2_loaded_actuators import Replay, csv_rows, save

GAINS = [.65, .7, .725, .75, .8]
TIME_CONSTANTS_MS = [0., 5., 10., 15.]


def aggregate(rows, repeats):
    selected = [row for row in rows if row['repeat'] in repeats]
    invalid = [row for row in selected if not row['valid']]
    return dict(windows=len(selected), valid_windows=len(selected)-len(invalid),
                complete=not invalid and bool(selected),
                score=float(np.mean([row['score'] for row in selected])) if selected and not invalid else None)


def evaluate(replay, name, config, duration):
    rows = []
    for window in replay.windows:
        row = dict(candidate=name, duration_s=duration, window=window['label'], repeat=window['repeat'])
        try:
            metrics, _, _, _, _ = replay.run(config, window, velocity='both', clearance=0.)
            row.update(valid=True, error='', **metrics)
        except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
            row.update(valid=False, error=str(error), score=None)
        rows.append(row)
    record = dict(candidate=name, config=config, duration_s=duration,
                  train=aggregate(rows, (1, 2)), validation=aggregate(rows, (3,)))
    return record, rows


def write_heterogeneous_csv(path, rows):
    keys = list(dict.fromkeys(key for row in rows for key in row))
    csv_rows(path, [{key: row.get(key) for key in keys} for row in rows])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--horizons', type=float, nargs='*', default=[.1, .5])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    candidate = json.loads(args.candidate.read_text())
    config = candidate.get('config', candidate)
    if set(config) != {'left_knee'}:
        raise ValueError('expected a left-knee-only fitted candidate')
    start = time.monotonic()
    replay = Replay(args.dataset)
    duration = float(replay.a['duration'])
    records, rows = [], []

    def run(name, parameters, runner=replay, seconds=duration):
        record, windows = evaluate(runner, name, parameters, seconds)
        records.append(record)
        rows.extend(windows)
        save(args.output/'progress.json', records)
        print(name, seconds, 'train', record['train'], 'validation', record['validation'],
              'elapsed_s', round(time.monotonic()-start, 2), flush=True)
        return record

    nominal = run('nominal', {})
    fitted = run('previous_fit', config)
    grid = []
    for gain in GAINS:
        for tau in TIME_CONSTANTS_MS:
            parameters = {'left_knee': dict(gain=gain, time_constant_ms=tau)}
            grid.append(run(f'gain_{gain:g}_tau_{tau:g}', parameters))

    valid_grid = [record for record in grid if record['train']['complete']]
    if not valid_grid:
        raise ValueError('no complete training grid points')
    minimum = min(record['train']['score'] for record in valid_grid)
    best = min(valid_grid, key=lambda record: record['train']['score'])
    valley = [record for record in valid_grid if record['train']['score'] <= minimum*1.05]
    for record in grid:
        record['train_fraction_above_grid_min'] = (record['train']['score']/minimum-1
            if record['train']['complete'] else None)
    profile = []
    for tau in TIME_CONSTANTS_MS:
        values = [record for record in valid_grid
                  if record['config']['left_knee']['time_constant_ms'] == tau]
        if values:
            selected = min(values, key=lambda record: record['train']['score'])
            profile.append(dict(time_constant_ms=tau,
                                minimizing_gain=selected['config']['left_knee']['gain'],
                                train_score=selected['train']['score'],
                                validation_score=selected['validation']['score'],
                                train_fraction_above_grid_min=selected['train']['score']/minimum-1))
    summary = dict(
        scope='conditional loss profile, not identified hardware constants or statistical confidence bounds',
        selection='grid minimum and per-tau profile selected using repetitions 1 and 2 only; original candidate unchanged',
        holdout='repetition 3 reported separately; same session, not independent blind validation',
        nuisance='both-foot no-slip root velocity, minimum sole vertex at z=0, zero bristle preload',
        gains=GAINS, time_constants_ms=TIME_CONSTANTS_MS,
        nominal=nominal, previous_fit=fitted, training_grid_minimum=best,
        descriptive_5percent_valley=[dict(config=record['config'], train_score=record['train']['score'],
                                        validation_score=record['validation']['score']) for record in valley],
        valley_warning='5 percent is a descriptive tolerance, not a confidence interval',
        time_constant_profile=profile,
        interpretation='A time constant near 5 ms should not be reported as precise hardware delay; examine the profile and nuisance/horizon sensitivity.')
    save(args.output/'profile.json', summary)

    motor_dof = int(replay.env.mj_model.joint('left_knee_motor_joint').dofadr[0])
    mechanical = []
    for field, increments, units in [
            ('dof_frictionloss', [2., 5., 10., 20.], 'Nm'),
            ('dof_damping', [1., 2., 5.], 'Nm*s/rad')]:
        values = getattr(replay.env.mj_model, field)
        native = float(values[motor_dof])
        try:
            for increment in increments:
                values[motor_dof] = native + increment
                name = f'left_knee_motor_{field}_add_{increment:g}'
                record = run(name, {})
                record['mechanical_hypothesis'] = dict(
                    joint='left_knee_motor_joint', dof=motor_dof, field=field,
                    native_value=native, increment=increment,
                    candidate_value=native+increment, units=units,
                    coordinate='native motor-side generalized coordinate',
                    motor_response='identity; no additional gain or lag')
                mechanical.append(record)
        finally:
            values[motor_dof] = native
    save(args.output/'mechanical_profile.json', dict(
        scope='alternative conditional loss hypotheses, not physically identified motor friction or damping',
        selection='training repetitions1,2 only; repetition3 reported separately',
        reference_nominal=nominal, reference_gain_candidate=fitted,
        parameters_restored=True, candidates=mechanical))
    csv_rows(args.output/'mechanical_grid.csv', [dict(
        candidate=record['candidate'], **record['mechanical_hypothesis'],
        train_score=record['train']['score'], validation_score=record['validation']['score'],
        train_complete=record['train']['complete'], validation_complete=record['validation']['complete'])
        for record in mechanical])

    horizons = []
    with np.load(args.dataset) as source:
        arrays = {key: source[key].copy() for key in source.files}
    for horizon in args.horizons:
        if not np.isfinite(horizon) or horizon <= 0:
            raise ValueError('horizons must be finite and positive')
        path = args.output/f'dataset_horizon_{horizon:g}.npz'
        np.savez_compressed(path, **{**arrays, 'duration': np.asarray(horizon)})
        extension = Replay(path)
        try:
            for name, parameters in [('nominal', {}), ('previous_fit', config)]:
                horizons.append(run(name, parameters, extension, horizon))
        finally:
            extension.a.close()
    save(args.output/'horizons.json', horizons)
    write_heterogeneous_csv(args.output/'windows.csv', rows)
    csv_rows(args.output/'grid.csv', [dict(
        gain=record['config']['left_knee']['gain'],
        time_constant_ms=record['config']['left_knee']['time_constant_ms'],
        train_score=record['train']['score'], validation_score=record['validation']['score'],
        train_complete=record['train']['complete'], validation_complete=record['validation']['complete'],
        train_fraction_above_grid_min=record['train_fraction_above_grid_min']) for record in grid])
    save(args.output/'provenance.json', dict(
        dataset=str(args.dataset.resolve()), candidate=str(args.candidate.resolve()),
        hashes={str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in (Path(__file__), args.dataset, args.candidate,
                             Path(__file__).with_name('identify_h2_loaded_actuators.py'))},
        elapsed_s=time.monotonic()-start, physics_timestep_s=replay.env.sim_dt,
        safety='offline MuJoCo only, no DDS, no default asset or policy modification'))
    replay.a.close()


if __name__ == '__main__':
    main()
