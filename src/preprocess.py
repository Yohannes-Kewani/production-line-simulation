import pandas as pd

def load_data(path):
    # Load dataset from CSV
    return pd.read_csv(path)
def summary(df, pred=None):
    obs = df.shape[0]
    Types = df.dtypes
    Counts = df.apply(lambda x: x.count())
    Min = df.min()
    Max = df.max()
    Uniques = df.apply(lambda x: x.unique().shape[0])
    Nulls = df.apply(lambda x: x.isnull().sum())
    print('Data shape:', df.shape)

    if pred is None:
        cols = ['Types', 'Counts', 'Uniques', 'Nulls', 'Min', 'Max']
        str = pd.concat([Types, Counts, Uniques, Nulls, Min, Max], axis = 1, sort=True)

    str.columns = cols
    print('___________________________\nData Types:')
    print(str.Types.value_counts())
    print('___________________________')
    return str
def null_values(df):
    # a percentage to show null values
    nv = pd.concat([df.isnull().sum(), 100*df.isnull().sum()/df.shape[0]],axis=1).rename(columns={0:'Missing_Records', 1:'Percentage %'})
    return nv[nv.Missing_Records>0].sort_values('Missing_Records', ascending=False)
def unique_columns(df):
    # columns with unique values
    unique_col=[]
    for col in df.columns:
        if df[col].nunique()==1:
            unique_col.append(col)
    return unique_col
# Remove collinear features
def remove_collinear_features(df, threshold=0.7):
    """ Removing collinear features helps to:
        1. Improve model interpretability: collinear features can make it difficult to interpret the effects of individual features on the target varibable.
        2. Reduce overfitting: collinear features can lead to overfitting, as the model may  learn to rely on redundant information resulting in poor new data predictions.
        3. Enhace model stability: removing collinear features can make the model more stable and less sensitive to small changes in the data.
         4.Improve computational efficiency: removing collinear features can reduce the number of features in the dataset, which can improve the computational efficiency of the model training process. """
    # calcualte the correlation matrix
    corr_matrix = df.corr()
    iters = range(len(corr_matrix.columns)-1)
    drop_cols = []
    for i in iters:
        for j in range(i+1):
            items = corr_matrix.iloc[j:j+1, i+1:i+2]
            col = items.columns
            row = items.index
            val = abs(items.values)
            if val >= threshold:
                
                print(col.values[0], "|", row.values[0], "|", round(val[0][0],2))
                drop_cols.append(col.values[0])
    drop=set(drop_cols)
    df = df.drop(columns=drop)
    return df

