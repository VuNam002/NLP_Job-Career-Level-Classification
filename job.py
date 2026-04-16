import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile
from sklearn.model_selection import GridSearchCV
from imblearn.over_sampling import RandomOverSampler, SMOTEN
import joblib

def filter_location(location):
    result = re.findall("\,\s[A-Z]{2}$", location)
    if len(result) != 0:
        return result[0][2:]
    else:
        return location


data = pd.read_excel("D:/NLP/Job/final_project.ods", engine="odf",dtype=str)
data = data.dropna(axis=0)
data["location"] = data["location"].apply(filter_location)

target = "career_level"
x = data.drop(target, axis=1)
y = data[target]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=100, stratify=y)

# ros = SMOTEN(random_state=0, k_neighbors=2, sampling_strategy={
#     "director_business_unit_leader": 500,
#     "specialist": 500,
#     "managing_director_small_medium_company": 500,
#     "bereichsleiter": 1000
# })
# print(y_train.value_counts())
# x_train, y_train = ros.fit_resample(x_train, y_train)
# print("-----------------")
# print(y_train.value_counts())

preprocessor = ColumnTransformer(transformers=[
    ("title_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "title"),
    ("location_ft", OneHotEncoder(handle_unknown="ignore"), ["location"]),
    ("des_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=0.01, max_df=0.95), "description"),
    ("function_ft", OneHotEncoder(handle_unknown="ignore"), ["function"]),
    ("industry_ft", TfidfVectorizer(stop_words="english", ngram_range=(1, 1)), "industry")
])

cls = Pipeline(steps=[
    ("preprocessor", preprocessor),   # (6458, 7990)
    # ("feature_selector", SelectKBest(chi2, k=300)),
    ("feature_selector", SelectPercentile(chi2, percentile=5)),
    ("model", RandomForestClassifier())
])

params = {
    # "model__n_estimators": [100, 200, 300],
    "model__criterion": ["gini", "entropy", "log_loss"],
    "feature_selector__percentile": [1, 5, 10]
}


grid_search = GridSearchCV(estimator=cls, param_grid=params, cv=4, scoring="recall_weighted", verbose=2)
grid_search.fit(x_train, y_train)
y_predicted = grid_search.predict(x_test)
print(classification_report(y_test, y_predicted))

joblib.dump(grid_search.best_estimator_, 'job_model.pkl')
print("Da luu mo minh vao file 'job_model.pkl'")
