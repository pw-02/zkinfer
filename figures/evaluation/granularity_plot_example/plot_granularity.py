"""Six-panel partition-granularity figure. Requires matplotlib and numpy.

python plot_granularity.py --demo --output illustrative_granularity
python plot_granularity.py --csv measured_results.csv --output partition_granularity
CSV contains one completed trial per row; times are in seconds.
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

MODELS = ['MNIST GAN', 'MobileNetV2', 'NanoGPT (4L)']
G = [1, 2, 4, 8]
FIELDS = ['workload','g','requests','trial','completion_time_s','completed_requests','data_kind']


def demo_rows():
    # Invented to demonstrate the layout. Not measurements, simulation, or forecasts.
    curves = {
        'MNIST GAN': {1:[8.2,7.7,9.1,13.5],16:[58,55,65,92]},
        'MobileNetV2': {1:[91,95,116,162],16:[750,805,980,1340]},
        'NanoGPT (4L)': {1:[3.2,3.5,4.2,5.4],16:[27,25,24,31]},
    }
    rng = np.random.default_rng(17)
    rows = []
    for model in MODELS:
        for requests in [1,16]:
            for g, center in zip(G,curves[model][requests]):
                factors=[1-rng.uniform(.025,.055),1,1+rng.uniform(.025,.055)]
                for trial,factor in enumerate(factors,1):
                    rows.append(dict(workload=model,g=g,requests=requests,trial=trial,
                        completion_time_s=round(center*60*factor,3),
                        completed_requests=requests,data_kind='illustrative'))
    return rows


def validate(rows):
    groups,seen,kinds = {},set(),set()
    for row in rows:
        model,g,n = row['workload'],int(row['g']),int(row['requests'])
        if model not in MODELS or g not in G or n not in [1,16]:
            raise ValueError(f'Unexpected workload/granularity/request count: {model}/{g}/{n}')
        key=(model,g,n,str(row['trial']))
        if key in seen: raise ValueError(f'Duplicate trial: {key}')
        seen.add(key)
        if int(row['completed_requests']) != n:
            raise ValueError(f'Incomplete batch: {key}. Report failures separately, not as a completed makespan.')
        value=float(row['completion_time_s'])/60
        if not np.isfinite(value) or value<=0: raise ValueError('Completion time must be finite and positive')
        kind=row['data_kind']
        if kind not in {'measured','illustrative'}: raise ValueError('data_kind must be measured or illustrative')
        kinds.add(kind)
        groups.setdefault((model,g,n),[]).append(value)
    if len(kinds)!=1: raise ValueError('Cannot mix illustrative data with measured results')
    for model in MODELS:
        for n in [1,16]:
            for g in G:
                if (model,g,n) not in groups: raise ValueError(f'Missing cell: {model}, g={g}, N={n}')
    return groups,kinds=={'illustrative'}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--demo',action='store_true')
    source.add_argument('--csv',type=Path)
    parser.add_argument('--output',type=Path,default=Path('partition_granularity'))
    args=parser.parse_args()
    if args.demo: rows=demo_rows()
    else:
        with args.csv.open(newline='',encoding='utf-8-sig') as f: rows=list(csv.DictReader(f))
    groups,illustrative=validate(rows)
    plt.rcParams.update({
        'font.family':'serif','font.size':9,'axes.titlesize':10,'axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,
        'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.65,
        'pdf.fonttype':42,'ps.fonttype':42,
    })
    fig,axes=plt.subplots(2,3,figsize=(7.1,4.25))
    color='#1768AC'
    for col,model in enumerate(MODELS):
        for row,n in enumerate([1,16]):
            ax=axes[row,col]
            values=[groups[model,g,n] for g in G]
            med=np.array([np.median(v) for v in values])
            low=med-np.array([min(v) for v in values])
            high=np.array([max(v) for v in values])-med
            x=np.arange(4)
            ax.plot(x,med,color=color,marker='o',markerfacecolor='white',
                    markeredgewidth=1,markersize=4,linewidth=1.2,zorder=3)
            repeated=np.array([len(v)>1 for v in values])
            if repeated.any():
                ax.errorbar(x[repeated],med[repeated],yerr=[low[repeated],high[repeated]],
                            fmt='none',ecolor=color,elinewidth=.8,capsize=2,zorder=2)
            ax.set_xticks(x,G)
            ax.set_xlim(-.15,3.15)
            ax.set_ylim(0,max(max(v) for v in values)*1.16)
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4,min_n_ticks=3))
            ax.grid(axis='y',color='#dddddd',linewidth=.5,zorder=0)
            ax.tick_params(axis='both',length=3,width=.6)
            if row==0: ax.set_title(model,pad=9)
            else: ax.set_xlabel(r'Operators per partition, $g$',labelpad=5)
            if col==0:
                ax.set_ylabel('Single-request\nlatency (min)' if row==0 else '16-request batch\nmakespan (min)',labelpad=8)
    repeats={len(v) for v in groups.values()}
    note=('Median of 3 trials; bars show min–max' if repeats=={3} else
          'Single trial per configuration' if repeats=={1} else
          'Median across trials; bars show min–max where repeated')
    handle=Line2D([0],[0],color=color,marker='o',markerfacecolor='white',linewidth=1.2,markersize=4)
    fig.legend([handle],['zkInfer  |  '+note],loc='upper center',bbox_to_anchor=(.54,.995),frameon=False)
    fig.subplots_adjust(left=.105,right=.985,top=.855,bottom=.17 if illustrative else .13,wspace=.36,hspace=.42)
    if illustrative:
        fig.text(.54,.025,'ILLUSTRATIVE DATA ONLY — NOT EXPERIMENTAL RESULTS',
                 ha='center',fontsize=8,color='#9B2525')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for ext in ['pdf','svg','png']:
        fig.savefig(args.output.with_suffix('.'+ext),dpi=300,
                    metadata={'Creator':'zkInfer granularity plot; '+('ILLUSTRATIVE DATA' if illustrative else 'measured input')})
    plt.close(fig)
    if args.demo:
        with args.output.with_suffix('.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows)
    print('Saved',args.output,'(ILLUSTRATIVE)' if illustrative else '(measured input)')


if __name__=='__main__': main()
