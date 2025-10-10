from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.data_utils import object_to_int
from preprocess import process_data

def process_training_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.30, stratify=y)

    num_cols = ["tenure", 'MonthlyCharges', 'TotalCharges']
    
    scaler = StandardScaler()
    X_train[num_cols] = scaler.fit(X_train[num_cols])    
    
    X_train[num_cols] = scaler.transform(X_train[num_cols])
    X_test[num_cols] = scaler.transform(X_test[num_cols])
    
    return X_train, X_test, y_train, y_test, scaler