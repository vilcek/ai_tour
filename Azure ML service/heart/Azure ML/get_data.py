
import pandas as pd
from sklearn.model_selection import train_test_split

path_on_compute = "/tmp/classification-automl/"
path_on_datastore = "classification-automl/"

def get_data():
    df_heart = pd.read_csv(path_on_compute + path_on_datastore + "heart.csv")
    df_heart_X = df_heart.drop(["target"], axis=1).values
    df_heart_y = df_heart["target"].values
    X_train, X_test, y_train, y_test = train_test_split(df_heart_X, df_heart_y, test_size = 0.2, random_state=123)
    
    return { "X": X_train, "y": y_train, "X_valid": X_test, "y_valid": y_test }
