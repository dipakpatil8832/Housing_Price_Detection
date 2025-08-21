import numpy as np
import pandas as pd
import os
import joblib
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler , OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import cross_val_score

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

def build_pipeline(num_attribs ,cat_attribs ):
    num_pipline= Pipeline([
        ("imputer" , SimpleImputer(strategy="mean")),
        ("scaler" , StandardScaler())
    ])

    # For categorical columan

    cat_pipline = Pipeline([
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Construct the full pipline

    full_pipline = ColumnTransformer([
        ("num",num_pipline,num_attribs),
        ("cat", cat_pipline,cat_attribs)
    ])

    return full_pipline

if not os.path.exists(MODEL_FILE):
    

    housing = pd.read_csv("housing.csv")
 
    housing["income_cat"] = pd.cut(housing["median_income"],
                            bins=[0.0,1.5,3.0,4.5,6.0,np.inf],
                            labels=[1,2,3,4,5])

    split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42)

    for train_index, test_index in split.split(housing, housing["income_cat"]):
        train_set = housing.loc[train_index].drop("income_cat", axis=1)
        test_set = housing.loc[test_index].drop("income_cat", axis=1)

        # Save test set for inference (without income_cat)
    test_set.to_csv("input.csv", index=False)

    # Prepare features and labels from the training set
    housing_labels = train_set["median_house_value"].copy()
    housing_features = train_set.drop("median_house_value", axis=1)  # no income_cat here


    num_attribs = housing_features.drop("ocean_proximity" , axis=1).columns.tolist()
    cat_attribs = ["ocean_proximity"]

    Pipeline = build_pipeline(num_attribs,cat_attribs)
    # print(housing_features)
    housing_prepared = Pipeline.fit_transform(housing_features)
    # print(housing_prepared)

    model = RandomForestRegressor(random_state=42)
    model.fit(housing_prepared ,housing_labels)

    joblib.dump(model,MODEL_FILE)
    joblib.dump(Pipeline,PIPELINE_FILE)
    print("model is trained . Congrats !!!!! ")

else:
    # Lets do inference
    model =joblib.load(MODEL_FILE)
    Pipeline = joblib.load(PIPELINE_FILE)

    input_data = pd.read_csv("input.csv")
    transformed_input = Pipeline.transform(input_data)
    predictions = model.predict(transformed_input)
    input_data["median_house_value"]=predictions
    input_data.to_csv("output.csv",index=False)
    print("Inference is complete, result saved to output.csv Enjoy !!!!")