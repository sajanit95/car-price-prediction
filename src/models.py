import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor

from src.features import CAT_FEATURES, NUM_FEATURES


def build_preprocessor(cat_impute="constant"):
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler())])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="Missing") if cat_impute == "constant"
         else SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10))])
    return ColumnTransformer([("num", numeric, NUM_FEATURES), ("cat", categorical, CAT_FEATURES)])


def make_pipeline(estimator, log_target=True, cat_impute="constant"):
    pipe = Pipeline([("preprocess", build_preprocessor(cat_impute)), ("model", estimator)])
    if log_target:
        return TransformedTargetRegressor(regressor=pipe, func=np.log1p, inverse_func=np.expm1)
    return pipe


def candidate_estimators(random_state=42):
    return {
        "Baseline (median)": DummyRegressor(strategy="median"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(random_state=random_state),
        "KNN": KNeighborsRegressor(n_neighbors=10),
        "DecisionTree": DecisionTreeRegressor(random_state=random_state),
        "RandomForest": RandomForestRegressor(n_estimators=150, max_features=0.3, min_samples_leaf=2,
                                              random_state=random_state, n_jobs=-1),
        "SVR": SVR(),
        "XGBoost": XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=6,
                                random_state=random_state, n_jobs=-1),
        "LightGBM": LGBMRegressor(n_estimators=400, learning_rate=0.05,
                                  random_state=random_state, verbose=-1),
    }


P = "regressor__model__"
SEARCH_SPACES = {
    "Ridge": {P + "alpha": loguniform(0.01, 100)},
    "KNN": {P + "n_neighbors": randint(3, 30), P + "weights": ["uniform", "distance"]},
    "DecisionTree": {P + "max_depth": [None, 10, 15, 20, 30], P + "min_samples_leaf": randint(1, 20)},
    "RandomForest": {P + "n_estimators": [100, 150, 200], P + "max_features": [0.2, 0.3, 0.5],
                     P + "min_samples_leaf": [1, 2, 4], P + "max_depth": [None, 20, 30]},
    "SVR": {P + "C": loguniform(1, 100), P + "gamma": ["scale", 0.01, 0.03, 0.1],
            P + "epsilon": [0.01, 0.05, 0.1]},
    "XGBoost": {P + "n_estimators": randint(300, 1500), P + "learning_rate": loguniform(0.02, 0.15),
                P + "max_depth": randint(3, 10), P + "min_child_weight": randint(1, 10),
                P + "subsample": uniform(0.7, 0.3), P + "colsample_bytree": uniform(0.5, 0.5),
                P + "reg_lambda": loguniform(0.1, 10)},
    "LightGBM": {P + "n_estimators": randint(300, 1500), P + "learning_rate": loguniform(0.02, 0.15),
                 P + "num_leaves": randint(15, 128), P + "min_child_samples": randint(5, 50),
                 P + "subsample": uniform(0.7, 0.3), P + "subsample_freq": [1],
                 P + "colsample_bytree": uniform(0.5, 0.5), P + "reg_lambda": loguniform(0.1, 10)},
}
