import pandas as pd
import glob
import matplotlib.pyplot as plt
from numpy import log10 as log
import numpy as np

#node, batch, deploy, mode helps in setting appropriate filenames to the generated png images.
node = 'node0'
batch = 'batch128'
deploy = 'deploy3'
mode = 'batch'

# groups .csv files on a single node based on different filename criteria.
files_batch32 = glob.glob('dstat-*-deploy-*-batch-32.csv')
files_batch64 = glob.glob('dstat-*-deploy-*-batch-64.csv')
files_batch128 = glob.glob('dstat-*-deploy-*-batch-128.csv')
files_batch256 = glob.glob('dstat-*-deploy-*-batch-256.csv')
files_deploy1 = glob.glob('dstat-*-deploy-1-batch-*.csv') 
files_deploy2 = glob.glob('dstat-*-deploy-2-batch-*.csv')
files_deploy3 = glob.glob('dstat-*-deploy-3-batch-*.csv')
#print(files_batch64)

#initialize dictionary for each system stat, to handle data from multiple files
receive={} #keys: filenames, values: receive data
send ={}
diskread={}
diskwrite={}
memory={}
cpu={}

# generate plot data for the files that meet a certain filename criteria (ex: files_batch128)
for f in files_batch128:
 df = pd.read_csv(f,sep=',')
 df = df[df['cpu_python3'].apply(lambda x : not ':' in str(x))] #this was not handled in prev pre-processing, so adding here
 df = df[df['cpu_python3'].apply(lambda x : not '-' in str(x))] #this was not handled in prev pre-processing, so adding here
 receive[f]=log(df['recv'].astype('float32')).replace([np.inf,-np.inf],0) #log10 is used for visualization purpose
 send[f]=log(df['send'].astype('float32')).replace([np.inf,-np.inf],0) #log10 is used for visualization purpose
 diskwrite[f]=df['diskwrite'].astype('float32')/1000 #scaled to give values in KB
 diskread[f]=df['diskread'].astype('float32')/1000 #scaled to give values in KB
 memory[f]=df['memory'].astype('float32')/1000000 #scaled to give values in MB
 print(f)
 cpu[f]=df['cpu_python3'].astype('float32')

#since multiple files could be of varying lengths, I find the minimum length of the series-min_l
min_l=1000000
for i in receive:
 print(receive[i].size)
 min_l = min(min_l,receive[i].size)

#stores receive data of length 'min_l' from each file
plt.figure()
for i in receive:
 receive[i]=receive[i].iloc[0:min_l]
 plt.plot(receive[i],label=i)

plt.title('Network Receive)')
plt.ylabel('log10(receive bytes)')
plt.xlabel('time')
plt.legend()
if (mode=='batch'):
 plt.savefig('Receive_'+batch+'_'+node+'.png')
else:
 plt.savefig('Receive_'+deploy+'_'+node+'.png') 

plt.figure()
for i in send:
 send[i]=send[i].iloc[0:min_l]
 plt.plot(send[i],label=i)

plt.title('Network Send)')
plt.ylabel('log10(send bytes)')
plt.xlabel('time')
plt.legend()

if (mode=='batch'):
 plt.savefig('Send_'+batch+'_'+node+'.png')
else:
 plt.savefig('Send_'+deploy+'_'+node+'.png')

plt.figure()
for i in diskwrite:
 diskwrite[i]=diskwrite[i].iloc[0:min_l]
 plt.plot(diskwrite[i],label=i)

plt.title('Disk Write)')
plt.ylabel('disk write (KB)')
plt.xlabel('time')
plt.legend()

if (mode=='batch'):
 plt.savefig('Diskwrite_'+batch+'_'+node+'.png')
else:
 plt.savefig('Diskwrite_'+deploy+'_'+node+'.png')

plt.figure()
for i in diskread:
 diskread[i]=diskread[i].iloc[0:min_l]
 plt.plot(diskread[i],label=i)

plt.title('Disk Read')
plt.ylabel('disk read(KB)')
plt.xlabel('time')
plt.legend()

if (mode=='batch'):
 plt.savefig('Diskread_'+batch+'_'+node+'.png')
else:
 plt.savefig('Diskread_'+deploy+'_'+node+'.png')

plt.figure()
for i in cpu:
 cpu[i]=cpu[i].iloc[0:min_l]
 plt.plot(cpu[i],label=i)

plt.title('CPU)')
plt.ylabel('cpu%')
plt.xlabel('time')
plt.legend()

if (mode=='batch'):
 plt.savefig('CPU_'+batch+'_'+node+'.png')
else:
 plt.savefig('CPU_'+deploy+'_'+node+'.png')

plt.figure()
for i in memory:
 memory[i]=memory[i].iloc[0:min_l]
 plt.plot(memory[i],label=i)

plt.title('Memory')
plt.ylabel('memory(MB)')
plt.xlabel('time')
plt.legend()

if (mode=='batch'):
 plt.savefig('Memory_'+batch+'_'+node+'.png')
else:
 plt.savefig('Memory_'+deploy+'_'+node+'.png')
