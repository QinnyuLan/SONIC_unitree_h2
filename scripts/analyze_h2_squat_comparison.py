#!/usr/bin/env python3
"""Offline descriptive comparison of H2 squat logs, not plant identification."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
LOGS = ROOT/'gear_sonic_deploy/logs/squat_compare'
RUNS = [('base01', 'Baseline', '#2478a6', 'dds_retry01'),
        ('id01', 'Identified ankles', '#188763', 'dds'),
        ('real01', 'Real H2', '#c44438', 'dds')]
JOINTS = [('left_hip_pitch', 0), ('left_knee', 3), ('left_ankle_roll', 4),
          ('left_ankle_pitch', 5), ('right_hip_pitch', 6), ('right_knee', 9),
          ('right_ankle_roll', 10), ('right_ankle_pitch', 11), ('waist_roll', 13), ('waist_pitch', 14)]


def columns(path, names, stop_ms=None):
    with path.open() as f:
        header = next(csv.reader(f))
        selected = [header.index(n) for n in names]
        if stop_ms is None:
            return np.loadtxt(f, delimiter=',', usecols=selected, ndmin=2)
        # Stop in the completed experiment, not at the changing end of a live recorder.
        time_index = header.index('time_monotonic_ms')
        def lines():
            for line in f:
                if float(line.split(',', time_index+1)[time_index]) > stop_ms:
                    break
                yield line
        return np.loadtxt(lines(), delimiter=',', usecols=selected, ndmin=2)


def stats(values):
    a = np.asarray(values)
    return dict(mean=float(np.mean(a)), rms=float(np.sqrt(np.mean(a*a))),
                min=float(np.min(a)), max=float(np.max(a)), p95_abs=float(np.percentile(np.abs(a),95)))


def write_csv(path, rows):
    with path.open('x', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rotation(wxyz):
    return Rotation.from_quat(wxyz[:, [1,2,3,0]]).as_euler('xyz', degrees=True)


def clock_plot(output, traces):
    fig, ax = plt.subplots(figsize=(7,4), layout='constrained')
    for name,label,color,_ in RUNS[:2]:
        for repeat in range(1,4):
            a=traces[f'{name}_{repeat}_physics']
            a=a[(a[:,0]>=0)&(a[:,0]<=8.5)]
            ax.plot(a[:,0],a[:,1],color=color,alpha=.75,label=label if repeat==1 else None)
    ax.plot([0,8.5],[0,8.5],'--',color='#707070',label='Synchronized clock (1:1)')
    ax.set(xlabel='Elapsed wall time (s)',ylabel='Elapsed simulation time (s)',
           title='Physics time versus wall time',xlim=(0,8.5),ylim=(0,9))
    ax.legend(loc='upper left',fontsize=9)
    ax.grid(alpha=.2)
    fig.savefig(output/'clock_comparison.png',dpi=170)
    plt.close(fig)


def load_deploy(path):
    p = path/'deploy'
    q = columns(p/'q.csv', ['time_monotonic_ms']+[f'q_{j}' for j in range(15)])
    dq = columns(p/'dq.csv', ['time_monotonic_ms']+[f'dq_{j}' for j in range(15)])
    quat = columns(p/'base_quat.csv', ['time_monotonic_ms','base_qw','base_qx','base_qy','base_qz'])
    play = columns(p/'motion_playing.csv', ['time_monotonic_ms','playing_0'])
    modes = columns(p/'encoder_mode.csv', ['time_monotonic_ms','encoder_mode_0'])
    with (p/'motion_name.csv').open() as f:
        motion_names = [row['motion_name'] for row in csv.DictReader(f)]
    assert len(q)==len(dq)==len(quat)==len(play)==len(modes)==len(motion_names)
    for a in (dq,quat,play,modes):
        np.testing.assert_allclose(q[:,0],a[:,0],atol=.002,rtol=0)
    assert set(motion_names)=={'squat_001__A359'}
    c = columns(p/'lowcmd_command.csv',['time_monotonic_ms','program_state','play']+
                [f'{prefix}_{j}' for prefix in ('q_target','dq_target','kp','kd','tau_ff') for j in range(15)])
    ci = np.searchsorted(c[:,0],q[:,0],side='right')-1
    assert ci.min()>=0
    selected = c[ci,3:].reshape(-1,5,15)
    target, vtarget, kp, kd, ff = selected.transpose(1,0,2)
    pd = kp*(target-q[:,1:])+kd*(vtarget-dq[:,1:])+ff
    mask = (play[:,1]>.5)&(c[ci,1]==2)&(c[ci,2]>.5)
    indices = np.flatnonzero(mask)
    groups = np.split(indices,np.flatnonzero(np.diff(indices)>1)+1)
    groups = [g for g in groups if len(g)>100]
    return dict(t=q[:,0],q=q[:,1:],dq=dq[:,1:],euler=rotation(quat[:,1:]),
                target=target,pd=pd,kp=kp,kd=kd,groups=groups,modes=modes[:,1],
                command_age_ms=q[:,0]-c[ci,0],c=c)


def main():
    global RUNS
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--baseline-run', default='base01')
    parser.add_argument('--baseline-label', default='Baseline')
    parser.add_argument('--identified-run', default='id01')
    parser.add_argument('--identified-label', default='Identified ankles')
    parser.add_argument('--real-run', default='real01')
    parser.add_argument('--baseline-dds', default=None)
    args=parser.parse_args()
    baseline_dds = args.baseline_dds or ('dds_retry01' if args.baseline_run == 'base01' else 'dds')
    RUNS = [(args.baseline_run, args.baseline_label, '#2478a6', baseline_dds),
            (args.identified_run, args.identified_label, '#188763', 'dds'),
            (args.real_run, 'Real H2', '#c44438', 'dds')]
    args.output.mkdir(parents=True,exist_ok=False)
    output=args.output
    all_data={}
    audit={}
    playback_rows=[]
    joint_rows=[]
    sim_rows=[]
    rise_rows=[]
    traces={}
    for name,label,color,dds_dir in RUNS:
        path=LOGS/name
        d=load_deploy(path)
        assert len(d['groups']) == 3, f'{name}: expected three complete playbacks'
        all_data[name]=d
        launch=json.loads((path/'deploy_launch.json').read_text())
        audit[name]=dict(launch=launch,metadata=json.loads((path/'deploy/metadata.json').read_text()),
                        dds_metadata=json.loads((path/dds_dir/'metadata.json').read_text()),
                        dds_directory=dds_dir)
        start=d['t'][d['groups'][0][0]]-3000
        end=d['t'][d['groups'][-1][-1]]+5000
        raw=columns(path/dds_dir/'lowstate.csv',['time_monotonic_ms','tick','mode_pr','crc_ok']+
                    [f'{prefix}_{j}' for prefix in ('q','dq','tau_est','motor_error','temperature_0','temperature_1') for j in range(15)],stop_ms=end+100)
        raw=raw[(raw[:,0]>=start)&(raw[:,0]<=end)]
        assert np.all(np.diff(raw[:,0])>=0)
        d['raw']=raw
        raw_cmd=columns(path/dds_dir/'lowcmd.csv',['time_monotonic_ms','mode_pr','crc_ok'],stop_ms=end+100)
        raw_cmd=raw_cmd[(raw_cmd[:,0]>=start)&(raw_cmd[:,0]<=end)]
        audit[name]['completed_window_dds']=dict(rows=len(raw),rows_command=len(raw_cmd),
            invalid_crc=int((raw[:,3]==0).sum()),mode_pr=np.unique(raw[:,2]).tolist(),
            receive_interval_ms=stats(np.diff(raw[:,0])),tick_delta=stats(np.diff(raw[:,1])),
            command_interval_ms=stats(np.diff(raw_cmd[:,0])),
            motor_errors=int(np.count_nonzero(raw[:,49:64])),max_temperature=float(raw[:,64:].max()))
        print(name,'DDS',audit[name]['completed_window_dds'],flush=True)
        truth=None
        if name != args.real_run:
            truth_names=['time_monotonic_ms','sim_time_s','elastic_enabled','command_age_ms','reset_epoch']
            truth_names += [f'{p}_{axis}' for p in ('root_pos_world','root_velocity_world','com_world','band_force_world') for axis in 'xyz']
            truth_names += [f'root_quat_wxyz_{axis}' for axis in 'wxyz']
            for side in ('left','right'):
                truth_names += [f'{side}_{p}_{axis}' for p in ('pos_world','velocity_world','force_world','cop_z0_world') for axis in 'xyz']
                truth_names += [f'{side}_contact_count',f'{side}_cop_valid']
            truth_names += [f'{p}_{j}' for p in ('q','dq','pd_requested','motor_requested','motor_ctrl_clipped','motor_force_applied','output_equivalent_motor_force','motor_clipped','extra_pitch_force') for j in range(15)]
            arr=columns(path/'sim/truth.csv',truth_names,stop_ms=end+100)
            truth={n:arr[:,i] for i,n in enumerate(truth_names)}
            d['truth']=truth
            audit[name]['sim_metadata']=json.loads((path/'sim/metadata.json').read_text())
            timing = audit[name]['sim_metadata'].get('realtime')
            if timing is not None:
                assert timing['valid'], f'{name}: invalid realtime capture'
                assert audit[name]['sim_metadata']['status'] == 'completed', f'{name}: incomplete capture'
        for repeat,g in enumerate(d['groups'],1):
            t0=d['t'][g[0]];t1=d['t'][g[-1]]+20
            rel=(d['t']-t0)/1000
            pre=(rel>=-3)&(rel<0)
            post=(rel>=8.5)&(rel<13.5)
            knee=np.rad2deg(d['q'][:,[3,9]].mean(axis=1))
            knee_speed=np.convolve(d['dq'][:,[3,9]].mean(axis=1),np.ones(5)/5,mode='same')
            extend=g[knee_speed[g]<np.deg2rad(-5)]
            row=dict(run=name,repeat=repeat,start_ms=t0,duration_wall_s=(t1-t0)/1000,
                     samples=len(g),encoder_modes=str(np.unique(d['modes'][g]).tolist()),
                     knee_peak_deg=float(knee[g].max()),pitch_peak_deg=float(d['euler'][g,1].max()),
                     pitch_min_deg=float(d['euler'][g,1].min()),pitch_pre_mean_deg=float(d['euler'][pre,1].mean()),
                     pitch_post_mean_deg=float(d['euler'][post,1].mean()),
                     ankle_left_pre_deg=float(np.rad2deg(d['q'][pre,5].mean())),
                     ankle_right_pre_deg=float(np.rad2deg(d['q'][pre,11].mean())),
                     command_age_p95_ms=float(np.percentile(d['command_age_ms'][g],95)),
                     command_age_max_ms=float(d['command_age_ms'][g].max()))
            playback_rows.append(row)
            for window,chosen in [('playback',g),('knee_extension',extend)]:
                for joint,j in JOINTS:
                    error=np.rad2deg(d['target'][chosen,j]-d['q'][chosen,j])
                    jr=dict(run=name,repeat=repeat,window=window,joint=joint,dds=j,
                            kp=float(np.median(d['kp'][chosen,j])),kd=float(np.median(d['kd'][chosen,j])))
                    for prefix,val in [('error_deg',error),('q_deg',np.rad2deg(d['q'][chosen,j])),
                                       ('target_deg',np.rad2deg(d['target'][chosen,j])),
                                       ('dq_deg_s',np.rad2deg(d['dq'][chosen,j])),('pd_nm',d['pd'][chosen,j])]:
                        jr.update({prefix+'_'+key:value for key,value in stats(val).items()})
                    joint_rows.append(jr)
            peaks,_=find_peaks(knee[g],prominence=30,distance=40)
            for cycle,p in enumerate(peaks,1):
                peak=g[p]
                candidates=g[(g>peak)&(knee[g]<15)]
                finish=int(candidates[0]) if len(candidates) else int(g[-1])
                rr=dict(run=name,repeat=repeat,cycle=cycle,peak_time_s=rel[peak],finish_time_s=rel[finish],
                        extension_duration_wall_s=rel[finish]-rel[peak],knee_peak_deg=knee[peak],
                        pelvis_pitch_finish_deg=d['euler'][finish,1])
                after = (rel >= rel[finish] + .3) & (rel < rel[finish] + .8)
                rr['postrise_pitch_mean_deg'] = float(d['euler'][after, 1].mean())
                for side,j in [('left',5),('right',11)]:
                    rr.update({side+'_q_finish_deg':np.rad2deg(d['q'][finish,j]),
                               side+'_target_finish_deg':np.rad2deg(d['target'][finish,j]),
                               side+'_pd_finish_nm':d['pd'][finish,j],
                               side+'_postrise_q_mean_deg':float(np.rad2deg(d['q'][after,j].mean()))})
                rise_rows.append(rr)
            print('PLAYBACK',row,flush=True)
            chosen=(rel>=-3)&(rel<=13.5)
            traces[f'{name}_{repeat}_joint']=np.column_stack((rel[chosen],knee[chosen],d['euler'][chosen,1],d['q'][chosen],d['target'][chosen],d['pd'][chosen]))
            if truth is not None:
                tt=truth['time_monotonic_ms']
                mask=(tt>=t0)&(tt<t1)
                assert len(np.unique(truth['reset_epoch'][mask])) == 1, f'{name}: reset during playback'
                xyz=lambda prefix:np.column_stack([truth[prefix+'_'+axis] for axis in 'xyz'])
                root=xyz('root_pos_world');lpos=xyz('left_pos_world');rpos=xyz('right_pos_world')
                quat=np.column_stack([truth['root_quat_wxyz_'+a] for a in 'wxyz'])
                yaw=np.deg2rad(rotation(quat)[np.flatnonzero(mask)[0],2]);forward=np.array([np.cos(yaw),np.sin(yaw),0])
                for key,vals in [('root_forward',root),('left_foot_forward',lpos),('right_foot_forward',rpos)]:
                    traces[f'{name}_{repeat}_{key}']=np.column_stack(((tt-t0)/1000,(vals-vals[mask][0])@forward))
                fz=truth['left_force_world_z']+truth['right_force_world_z']
                simelapsed=truth['sim_time_s'][mask][-1]-truth['sim_time_s'][mask][0]
                wallelapsed=(tt[mask][-1]-tt[mask][0])/1000
                sr=dict(run=name,repeat=repeat,duration_sim_s=simelapsed,realtime_factor=simelapsed/wallelapsed,
                        elastic_any=int(np.any(truth['elastic_enabled'][mask])),band_force_max_n=float(np.abs(xyz('band_force_world')[mask]).max()),
                        root_net_forward_m=float(((root[mask][-1]-root[mask][0])@forward)),
                        root_forward_range_m=float(np.ptp(root[mask]@forward)),
                        left_foot_net_forward_m=float((lpos[mask][-1]-lpos[mask][0])@forward),
                        right_foot_net_forward_m=float((rpos[mask][-1]-rpos[mask][0])@forward),
                        left_unloaded_fraction=float(np.mean(truth['left_force_world_z'][mask]<20)),
                        right_unloaded_fraction=float(np.mean(truth['right_force_world_z'][mask]<20)),
                        min_total_fz_n=float(fz[mask].min()),max_total_fz_n=float(fz[mask].max()),
                        left_load_share_mean=float(np.mean(truth['left_force_world_z'][mask]/np.maximum(1,fz[mask]))))
                for joint,j in JOINTS:
                    sr[joint+'_clip_fraction']=float(np.mean(truth[f'motor_clipped_{j}'][mask]))
                for side,j in [('left',5),('right',11)]:
                    sr[side+'_extra_pitch_p95_nm']=float(np.percentile(np.abs(truth[f'extra_pitch_force_{j}'][mask]),95))
                    sr[side+'_extra_pitch_max_nm']=float(np.max(np.abs(truth[f'extra_pitch_force_{j}'][mask])))
                    mismatch=truth[f'output_equivalent_motor_force_{j}'][mask]-truth[f'pd_requested_{j}'][mask]
                    sr[side+'_mapped_effort_error_rms_nm']=float(np.sqrt(np.mean(mismatch*mismatch)))
                sim_rows.append(sr)
                print('SIM',sr,flush=True)
                keep=(tt>=t0-3000)&(tt<t1+5000)
                traces[f'{name}_{repeat}_physics']=np.column_stack(((tt[keep]-t0)/1000,
                    truth['sim_time_s'][keep]-np.interp(t0,tt,truth['sim_time_s']),fz[keep],
                    truth['extra_pitch_force_5'][keep],truth['extra_pitch_force_11'][keep]))
    audit['hashes_equal']=all(audit[n]['launch']['files']==audit[args.baseline_run]['launch']['files'] for n,_,_,_ in RUNS)
    assert audit['hashes_equal'], 'model/reference/config mismatch between runs'
    (output/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    write_csv(output/'playbacks.csv',playback_rows)
    write_csv(output/'joint_statistics.csv',joint_rows)
    write_csv(output/'simulation_statistics.csv',sim_rows)
    write_csv(output/'rise_events.csv',rise_rows)
    np.savez_compressed(output/'traces.npz',**traces)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    clock_plot(output,traces)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained',sharex=True)
    grid=np.arange(0,8.46,.02)
    for name,label,color,_ in RUNS:
        d=all_data[name]
        for ax,key,j,scale in [(axes[0,0],'q',[3,9],180/np.pi),(axes[0,1],'euler',1,1),
                               (axes[1,0],'q',5,180/np.pi),(axes[1,1],'q',11,180/np.pi)]:
            curves=[]
            for g in d['groups']:
                y=d[key][g][:,j]
                if y.ndim==2:y=y.mean(axis=1)
                curves.append(np.interp(grid,(d['t'][g]-d['t'][g[0]])/1000,y*scale))
            curves=np.array(curves)
            ax.plot(grid,np.median(curves,axis=0),color=color,label=label)
            ax.fill_between(grid,curves.min(axis=0),curves.max(axis=0),color=color,alpha=.12)
    for ax,title in zip(axes.flat,['Mean knee angle','Pelvis IMU pitch','Left ankle pitch (measured)','Right ankle pitch (measured)']):
        ax.set(title=title,ylabel='Angle (deg)',xlabel='Wall time since playback (s)');ax.grid(alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, fontsize=9, frameon=False)
    fig.suptitle('Same policy/reference/gains | median and range of 3 playbacks\nPlayback-aligned time; independent closed-loop commands',fontsize=12)
    fig.savefig(output/'measured_comparison.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,3,figsize=(13,6),layout='constrained',sharex=True)
    for col,(name,label,color,_) in enumerate(RUNS):
        d=all_data[name];g=d['groups'][0];t=(d['t'][g]-d['t'][g[0]])/1000
        for row,j in enumerate([5,11]):
            axes[row,col].plot(t,np.rad2deg(d['target'][g,j]),'--',color='#707070',label='Published q target')
            axes[row,col].plot(t,np.rad2deg(d['q'][g,j]),color=color,label='Measured q')
            axes[row,col].set(title=f'{label}: {"left" if row==0 else "right"}',ylabel='Angle (deg)',xlabel='Wall time (s)',ylim=(-120,140))
            axes[row,col].grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.suptitle('First playback: actual LowCmd versus ankle feedback\nVirtual PD targets request torque; target-minus-q is not pure servo delay',fontsize=12)
    fig.savefig(output/'ankle_inputs_outputs.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for name,label,color,_ in RUNS[:2]:
        for repeat in range(1,4):
            ph=traces[f'{name}_{repeat}_physics'];fw=traces[f'{name}_{repeat}_root_forward']
            axes[0,0].plot(ph[:,0],ph[:,1],color=color,alpha=.5,label=label if repeat==1 else None)
            axes[0,1].plot(fw[:,0],fw[:,1]*100,color=color,alpha=.5)
            axes[1,0].plot(ph[:,0],ph[:,2],color=color,alpha=.4)
            axes[1,1].plot(ph[:,0],ph[:,4],color=color,alpha=.5)
    axes[0,0].plot([0,8.5],[0,8.5],'--',color='#707070',label='Real-time target')
    for ax,title,ylabel in zip(axes.flat,['Physics clock','Pelvis forward displacement','Total foot ground force','Additional right ankle friction'],
                               ['Simulation elapsed (s)','Displacement (cm)','Vertical force (N)','Output-DOF force (Nm)']):
        ax.set(title=title,ylabel=ylabel,xlabel='Wall time since playback (s)',xlim=(0,8.5));ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=9)
    fig.suptitle('Simulation solver diagnostics | all 3 playbacks per model\nElastic support disabled; displacement relative to each playback start',fontsize=12)
    fig.savefig(output/'simulation_diagnostics.png',dpi=160);plt.close(fig)
    print('OUTPUT',output)


if __name__=='__main__':
    main()
