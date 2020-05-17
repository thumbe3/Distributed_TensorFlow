import pandas as pd
import sys
df = pd.read_csv(sys.argv[1],sep=',',names=['time','usr','sys','idl','wai','hiq','siq','cpu_python3','diskread','diskwrite','recv','send','memory'])
df['memory']=df['memory'].map(lambda x: x.split('/')[1][:-1])
df['cpu_python3']=df['cpu_python3'].map(lambda x:x.split('/')[1][0:-1])
df.to_csv(sys.argv[1],columns=['time','recv','send','diskwrite','diskread','memory','cpu_python3'])
