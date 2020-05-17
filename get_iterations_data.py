import pandas as pd
import sys

def main():
    filename1 = sys.argv[1]
    filename2 = sys.argv[2]
    filename3 = sys.argv[3]
    df = pd.read_csv(filename1, sep = ' ', header=None)
    df.columns = ['Task','Red1','Iterations', 'red2']
    foo = lambda x: pd.Series(x.split('.')[0].split('-')[1:])
    stats = df['Task'].apply(foo)
    stats.columns = ['Worker', 'Deploy-Mode', 'Batch_Size']
    iterations = pd.DataFrame()
    iterations['Iterations'] = df['Iterations']
    merged = pd.concat([stats, iterations], axis=1)
    dk = pd.read_csv(filename2, sep = ' ', header=None)
    dk = dk.drop([1,2,3,4,6], axis=1)
    dk.columns = ['Task', 'Worker', 'Total time(seconds)']
    merged['Total time(seconds)'] = dk['Total time(seconds)']
    da = pd.read_csv(filename3,sep = ' ', header=None)
    da.columns = ['Task', 'redundant','Accuracy(%)']
    da['Accuracy(%)'] = da['Accuracy(%)'].apply(lambda x: x*100)
    merged['Accuracy(%)'] = da['Accuracy(%)']
    merged['Batch_Size'] = merged['Batch_Size'].astype(int)
    merged = merged.sort_values(['Worker', 'Deploy-Mode','Batch_Size'])
    merged.to_csv(filename1.split('_')[0]+'time_iterations_accuracy_parsed.csv', index=False)

if __name__ =="__main__":
    main()
