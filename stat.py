#%%
from datetime import datetime, timedelta
from multiprocessing import Pool, cpu_count

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams["figure.figsize"] = (16, 12)

raw = pd.read_excel('US Rate  Stock fed fund Data_1960 to 2023.xlsx')

def process(key):
    tmp = raw[map_dict[key]].dropna().rename(columns={map_dict[key][0]: 'date',
                                                      map_dict[key][1]: key})
    tmp = tmp.set_index(pd.DatetimeIndex(tmp['date'])).drop(columns=['date'])
    return tmp

if __name__ == '__main__':
    map_dict = {
        '10YR': ['date', 'USGG10YR Index'],
        'USGG10YR': ['date', 'Unnamed: 2'],
        'USGG10YR_adj': ['date', '含息報酬'],
        'SPX': ['date.1', 'SPX Index'],
        'SPX_adj': ['date.2', 'SPX Index含息'],
        'FDTR': ['date.4', 'FDTR Index'],
        'PCE': ['date.5', 'PCE CYOY Index']
    }
    
    p = Pool(int(cpu_count() * 0.9))
    result = p.map(process, map_dict.keys())
    p.close()
    p.join()
    
    df = pd.concat(result, axis=1)
    df['PCE'] = df['PCE'].fillna(method='ffill')
    df = df.dropna()

    total = df.copy()
    total = total.drop(columns=['USGG10YR', 'USGG10YR_adj', 'SPX_adj'])
    # %%
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax3 = ax1.twinx()
    ax4 = ax1.twinx()
    sns.lineplot(data=total, x=total.index, y='10YR', color="black", ax=ax1)
    sns.lineplot(data=total, x=total.index, y='SPX', color="red", ax=ax2)
    sns.lineplot(data=total, x=total.index, y='FDTR', color="blue", ax=ax3)
    sns.lineplot(data=total, x=total.index, y='PCE', color="green", ax=ax4)
    ax1.set_ylabel('')
    ax2.set_ylabel('')
    ax3.set_ylabel('')
    ax4.set_ylabel('')
    ax1.set_yticks([])
    ax2.set_yticks([])
    ax3.set_yticks([])
    ax4.set_yticks([])
    plt.show()
    # %%
    df = df.drop(columns=['10YR'])
    df['SPX'] = (np.log(df['SPX'])).diff(1)
    df['SPX_adj'] = (np.log(df['SPX_adj'])).diff(1)
    df = df.dropna()

    # 最後一次升息日
    hike_stop = df[df['FDTR'].diff(1)!=0]['FDTR']
    hike_stop = hike_stop[(hike_stop.diff(1)>0)&(hike_stop.diff(-1)>0)].index
    hike_stop = list(hike_stop)
    hike_stop.append(df.index[-1])
    # %%
    data = []
    for hs in hike_stop:
        tmp_idx = list(df.index).index(hs)
        if tmp_idx>252:
            tmp = df.reset_index().iloc[tmp_idx-252:tmp_idx+252]
        else:
            tmp = df.reset_index().iloc[:tmp_idx+252]
        tmp['t'] = tmp.index
        tmp['t'] -= tmp_idx
        
        tmp['SPX_p_3m'] = np.exp(tmp[(tmp.t < 0) & (tmp.t >= -60)]['SPX_adj'].sum())-1
        tmp['USGG10YR_p_3m'] = tmp[(tmp.t < 0) & (tmp.t >= -60)]['USGG10YR_adj'].sum()
        tmp['SPX_p_6m'] = np.exp(tmp[(tmp.t < 0) & (tmp.t >= -120)]['SPX_adj'].sum())-1
        tmp['USGG10YR_p_6m'] = tmp[(tmp.t < 0) & (tmp.t >= -120)]['USGG10YR_adj'].sum()
        tmp['SPX_p_1Y'] = np.exp(tmp[tmp.t < 0]['SPX_adj'].sum())-1
        tmp['USGG10YR_p_1Y'] = tmp[tmp.t < 0]['USGG10YR_adj'].sum()

        tmp['SPX_f_3m'] = np.exp(tmp[(tmp.t > 0) & (tmp.t <= 60)]['SPX_adj'].sum())-1
        tmp['USGG10YR_f_3m'] = tmp[(tmp.t > 0) & (tmp.t <= 60)]['USGG10YR_adj'].sum()
        tmp['SPX_f_6m'] = np.exp(tmp[(tmp.t > 0) & (tmp.t <= 120)]['SPX_adj'].sum())-1
        tmp['USGG10YR_f_6m'] = tmp[(tmp.t > 0) & (tmp.t <= 120)]['USGG10YR_adj'].sum()
        tmp['SPX_f_1y'] = np.exp(tmp[tmp.t > 0]['SPX_adj'].sum())-1
        tmp['USGG10YR_f_1y'] = tmp[tmp.t > 0]['USGG10YR_adj'].sum()
        
        tmp[['SPX', 'SPX_adj']] = np.exp(tmp[['SPX', 'SPX_adj']].cumsum())-1
        tmp[['USGG10YR', 'USGG10YR_adj']] = tmp[['USGG10YR', 'USGG10YR_adj']].cumsum()
        tmp['hike_stop'] = hs
        tmp['pce_chg_before'] = tmp[tmp.t == 0]['PCE'].values[0] - tmp.head(1)['PCE'].values[0]
        tmp['pce_chg_after'] = tmp.tail(1)['PCE'].values[0] - tmp[tmp.t == 0]['PCE'].values[0]
        tmp['pce_stop'] = tmp[tmp.t == 0]['PCE'].values[0]
        data.append(tmp)
    data = pd.concat(data).reset_index(drop=True)
    data['hike_stop'] = data['hike_stop'].dt.strftime('%Y/%m')
    data['pce_before_sign'] = np.sign(data['pce_chg_before'])
    data['pce_after_sign'] = np.sign(data['pce_chg_after'])
    # %%
    for y in ['USGG10YR_adj', 'SPX_adj']:
        g = sns.relplot(data=data,
                        x="t", y=y, col="hike_stop", hue="hike_stop",
                        kind="line", palette="dark:salmon_r", linewidth=2, zorder=5,
                        col_wrap=5, height=2, aspect=1.5, legend=False)
        for hs, ax in g.axes_dict.items():
            ax2 = ax.twinx()
            ax2.set_ylim(0, 11)
            # Add the title as an annotation within the plot
            ax.text(.1, .85, hs, transform=ax.transAxes, fontweight="bold")
            # Plot every year's time series in the background
            sns.lineplot(data=data, x="t", y=y, units="hike_stop",
                        estimator=None, color=".7", linewidth=.75, ax=ax)
            sns.lineplot(data=data[data.hike_stop==hs], x="t", y='PCE',
                        estimator=None, color="red", linewidth=.75, ax=ax2)
            ax.axvline(x=0, ymin=-1, ymax=1, color=".5")
            ax2.set_ylabel('')

        # Reduce the frequency of the x axis ticks
        ax.set_xticks(ax.get_xticks()[::2])

        # Tweak the supporting aspects of the plot
        g.set_titles("")
        g.set_axis_labels("", y)
        g.tight_layout()
    # %%
    data.groupby('hike_stop')[['hike_stop', 'pce_stop', 'pce_chg_before', 'pce_chg_after']].tail(1).set_index('hike_stop').rename(columns={'pce_stop': '停止時PCE',
                                                                                                                                           'pce_chg_before': '過去1年變化',
                                                                                                                                           'pce_chg_after': '未來1年變化'})
    # %%
    data.groupby('hike_stop')[['hike_stop', 'pce_before_sign', 'pce_after_sign', 'SPX_p_1Y', 'SPX_p_6m', 'SPX_p_3m', 'SPX_f_3m', 'SPX_f_6m', 'SPX_f_1y']].tail(1).set_index('hike_stop')
    # %%
    # 第一次降息
    cut_start = df[df['FDTR'].diff(1)!=0]['FDTR']
    cut_start = cut_start[(cut_start.diff(1)<0)&(cut_start.diff(1).shift(1)>0)].index
    # %%
    data2 = []
    for cs in cut_start:
        tmp_idx = list(df.index).index(cs)
        if tmp_idx>252:
            tmp = df.reset_index().iloc[tmp_idx-252:tmp_idx+252]
        else:
            tmp = df.reset_index().iloc[:tmp_idx+252]
        tmp['t'] = tmp.index
        tmp['t'] -= tmp_idx
        
        tmp['SPX_p_3m'] = np.exp(tmp[(tmp.t < 0) & (tmp.t >= -60)]['SPX_adj'].sum())-1
        tmp['USGG10YR_p_3m'] = tmp[(tmp.t < 0) & (tmp.t >= -60)]['USGG10YR_adj'].sum()
        tmp['SPX_p_6m'] = np.exp(tmp[(tmp.t < 0) & (tmp.t >= -120)]['SPX_adj'].sum())-1
        tmp['USGG10YR_p_6m'] = tmp[(tmp.t < 0) & (tmp.t >= -120)]['USGG10YR_adj'].sum()
        tmp['SPX_p_1Y'] = np.exp(tmp[tmp.t < 0]['SPX_adj'].sum())-1
        tmp['USGG10YR_p_1Y'] = tmp[tmp.t < 0]['USGG10YR_adj'].sum()

        tmp['SPX_f_3m'] = np.exp(tmp[(tmp.t > 0) & (tmp.t <= 60)]['SPX_adj'].sum())-1
        tmp['USGG10YR_f_3m'] = tmp[(tmp.t > 0) & (tmp.t <= 60)]['USGG10YR_adj'].sum()
        tmp['SPX_f_6m'] = np.exp(tmp[(tmp.t > 0) & (tmp.t <= 120)]['SPX_adj'].sum())-1
        tmp['USGG10YR_f_6m'] = tmp[(tmp.t > 0) & (tmp.t <= 120)]['USGG10YR_adj'].sum()
        tmp['SPX_f_1y'] = np.exp(tmp[tmp.t > 0]['SPX_adj'].sum())-1
        tmp['USGG10YR_f_1y'] = tmp[tmp.t > 0]['USGG10YR_adj'].sum()
        
        tmp[['SPX', 'SPX_adj']] = np.exp(tmp[['SPX', 'SPX_adj']].cumsum())-1
        tmp[['USGG10YR', 'USGG10YR_adj']] = tmp[['USGG10YR', 'USGG10YR_adj']].cumsum()
        tmp['cut_start'] = cs
        tmp['pce_chg_before'] = tmp[tmp.t == 0]['PCE'].values[0] - tmp.head(1)['PCE'].values[0]
        tmp['pce_chg_after'] = tmp.tail(1)['PCE'].values[0] - tmp[tmp.t == 0]['PCE'].values[0]
        tmp['pce_stop'] = tmp[tmp.t == 0]['PCE'].values[0]
        data2.append(tmp)
    data2 = pd.concat(data2).reset_index(drop=True)
    data2['cut_start'] = data2['cut_start'].dt.strftime('%Y/%m')
    data2['pce_before_sign'] = np.sign(data2['pce_chg_before'])
    data2['pce_after_sign'] = np.sign(data2['pce_chg_after'])
    # %%
    tmp = data2.groupby('cut_start')[['cut_start', 'pce_stop', 'pce_before_sign', 'pce_after_sign', 'SPX_p_1Y', 'SPX_p_6m', 'SPX_p_3m', 'SPX_f_3m', 'SPX_f_6m', 'SPX_f_1y']].tail(1).set_index('cut_start')
    tmp[['SPX_p_1Y', 'SPX_p_6m', 'SPX_p_3m', 'SPX_f_3m', 'SPX_f_6m', 'SPX_f_1y']]*=100
    tmp.round(2)
