from utils.data_utils import object_to_int
from sklearn.preprocessing import StandardScaler

def process_data(df, scaler):
    df = df.apply(lambda x: object_to_int(x))

    X = df.drop(columns = ['Churn'])
    y = df['Churn'].values

    num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']

    X[num_cols] = scaler.transform(X[num_cols])

    return X, y