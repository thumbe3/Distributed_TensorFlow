import pandas as pd
import pandas as pd
import sys
import re
import matplotlib.pyplot as plt


def main():
    filename = sys.argv[1]
    df = pd.read_csv(filename, sep='|', header=1)
    df.columns = [x.strip() for x in list(df.columns)]
    foo = lambda x: pd.Series([int(j) for j in re.sub(' +', ' ', x).strip().split(' ')])
    rev = df['usr sys idl wai hiq siq'].apply(foo)
    rev.columns = ['usr', 'sys', 'idl', 'wai', 'hiq', 'siq']
    foo = lambda x: pd.Series([j for j in x.split()])
    rw = df['read  writ'].apply(foo)
    rw.columns = ['read(KB)', 'write(KB)']
    foo = lambda x: pd.Series([j for j in x.split()])
    rs = df['recv  send'].apply(foo)
    rs.columns = ['receive(KB)', 'send(KB)']
    foo = lambda x: pd.Series([j for j in x.split()])
    cpu = df['cpu process'].apply(foo)
    cpu.columns = ['process', 'cpu%']
    foo = lambda x: pd.Series([j for j in x.split()])
    mem = df['memory process'].apply(foo)
    mem.columns = ['process_1', 'memory(KB)']
    time = pd.DataFrame()
    time['time'] = df['time']
    merged = pd.concat([time, rev, cpu, rw, rs, mem], axis=1)
    relevant = merged[merged['process'] == 'python3'].copy()
    memdict = {'k': 1000, 'b': 1, 'm': 1000000}
    relevant['write(KB)'] = relevant['write(KB)'].apply(
        lambda x: (int(x[0:-1]) * memdict[x[-1].lower()]) / 1000 if x != '0' else 0)
    relevant['receive(KB)'] = relevant['receive(KB)'].apply(
        lambda x: (int(x[0:-1]) * memdict[x[-1].lower()]) / 1000 if x != '0' else 0)
    relevant['send(KB)'] = relevant['send(KB)'].apply(
        lambda x: (int(x[0:-1]) * memdict[x[-1].lower()]) / 1000 if x != '0' else 0)
    relevant['memory(KB)'] = relevant['memory(KB)'].apply(
        lambda x: (int(x[0:-1]) * memdict[x[-1].lower()]) / 1000 if x != '0' else 0)
    for col in ['cpu%', 'read(KB)', 'write(KB)', 'receive(KB)', 'send(KB)', 'memory(KB)']:
        relevant[col] = relevant[col].astype(float)
    plt.figure()
    ax = relevant['receive(KB)'].plot()
    ax.set_title('Receive_plot')
    ax.set_ylabel('Receive data in KB')
    ax.set_xlabel('Time')
    ax.get_figure().savefig(filename+'_' + 'Receive_plot'+'.png')

    plt.figure()
    ax = relevant['send(KB)'].plot()
    ax.set_title('Send_plot')
    ax.set_ylabel('Send data in KB')
    ax.set_xlabel('Time')
    ax.get_figure().savefig(filename+'_' + 'Send_plot'+'.png')

    plt.figure()
    ax = relevant['write(KB)'].plot()
    ax.set_title('Write_plot')
    ax.set_ylabel('Memory in KB')
    ax.set_xlabel('Time')
    ax.get_figure().savefig(filename+'_' + 'Write_plot'+'.png')

    plt.figure()
    ax = relevant['memory(KB)'].plot()
    ax.set_title('Memory_plot')
    ax.set_ylabel('Memory in KB')
    ax.set_xlabel('Time')
    ax.get_figure().savefig(filename+'_' + 'Memory_plot'+'.png')

    plt.figure()
    ax = relevant['cpu%'].plot()
    ax.set_title('CPU_plot')
    ax.set_ylabel('%CPU')
    ax.set_xlabel('Time')
    ax.get_figure().savefig(filename+'_' + 'CPU_plot'+'.png')

    print('Plots generated')


if __name__ == '__main__':
    main()