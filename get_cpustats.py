import os
import pandas as pd
import re
import sys
import shutil

def main():
    directory = sys.argv[1]
    files = os.listdir(directory)
    output_df = pd.DataFrame(columns = ['Deploy_Mode', 'Batch_Size','Worker','cpu%',
    'read(KB)', 'write(KB)', 'receive(KB)', 'send(KB)', 'memory(KB)'])
    index = 0
    for file in files:
        print(file)
        df = pd.read_csv(directory + '/' +  file, sep='|', header=2).dropna()
        df.columns = [x.strip() for x in list(df.columns)]
        df['cpu process'] =df['cpu process'].apply(lambda x: x.strip())
        df = df.drop(df.loc[df['cpu process']==''].index)
        foo = lambda x: pd.Series([x.split('python3')[1]] if 'python3' in x else None)
        df['cpu process'] = df['cpu process'].apply(foo)
        df = df.dropna()
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
        mem = df['memory process'].apply(foo)
        mem.columns = ['process', 'memory(KB)']
        time = pd.DataFrame()
        time['time'] = df['time']
        merged = pd.concat([time, rev,rw, rs, mem], axis=1)
        merged['CPU(%)'] = df['cpu process']
        relevant = merged[merged['process'] == 'python3'].copy()
        memdict = {'k': 1000, 'b': 1, 'm': 1000000}
        relevant['write(KB)'] = relevant['write(KB)'].apply(lambda x: (int(x[0:-1]) * memdict[x[-1].lower()]) / 1000 if x != '0' else 0)
        relevant['read(KB)'] = relevant['read(KB)'].apply(lambda x: (int(x[0:-1]) * memdict[x[-1].lower()]) / 1000 if x != '0' else 0)
        relevant['receive(KB)'] = relevant['receive(KB)'].apply(lambda x: (int(x[0:-1]) * memdict[x[-1].lower()]) / 1000 if x != '0' else 0)
        relevant['send(KB)'] = relevant['send(KB)'].apply(lambda x: (int(x[0:-1]) * memdict[x[-1].lower()]) / 1000 if x != '0' else 0)
        relevant['memory(KB)'] = relevant['memory(KB)'].apply(lambda x: (int(x[0:-1]) * memdict[x[-1].lower()]) / 1000 if x != '0' else 0)
        if 'single' in file:
            deploy_mode = file.split('_')[1]
            batch_size = file.split('_')[2]
            worker = 'single'
        else:
            print('heerrerere')
            deploy_mode = file.split('_')[2]
            batch_size = file.split('_')[3]
            worker = file.split('_')[1]
        tempstat = [deploy_mode,batch_size, worker]
        print(tempstat)
        cols = ['CPU(%)', 'read(KB)', 'write(KB)', 'receive(KB)', 'send(KB)', 'memory(KB)']

        for i in range(len(cols)) :
            relevant[cols[i]] = relevant[cols[i]].astype(float)
            tempstat.append(max(relevant[cols[i]]))
        output_df.loc[index] = tempstat
        index +=1
        dir = './CSV_results/'
        if not os.path.exists(dir):
            os.mkdir(dir)
        relevant.to_csv(dir + file + worker+ '_' + 'cpumem.csv', index=False)
    #print(output_df)
    output_df['Batch_Size'] = output_df['Batch_Size'].astype(int)
    output_df = output_df.sort_values(['Deploy_Mode','Worker','Batch_Size'])
    output_df.to_csv(dir+directory.replace('/','').replace('.','')+'cpumem_summary.csv', index=False)
if __name__ == "__main__":
    main()
