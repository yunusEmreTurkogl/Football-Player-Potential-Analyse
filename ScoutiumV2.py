import pandas as pd
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import xgboost
import lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

warnings.simplefilter(action='ignore', category=Warning)

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', '{:.3f}'.format)

############################################################################################################3



attributes = pd.read_csv("machine_learning/datasets/scoutium_attributes.csv", sep=";")
attributes.head()

attributes = pd.read_csv("scoutium_attributes.csv", sep=";")
attributes.head()

potential_labels = pd.read_csv("machine_learning/datasets/scoutium_potential_labels.csv", sep=";")
potential_labels.head()



merge_columns = ["task_response_id", 'match_id', 'evaluator_id', "player_id"]

df = pd.merge(attributes, potential_labels, on= merge_columns)

df.head()



df = df.loc[~(df["position_id"] == 1)]
df["position_id"].value_counts()


df["position_id"].value_counts()
df = df[df["position_id"] != 1]



df.head()
df["potential_label"].value_counts() / df.shape[0]

df = df.loc[~(df["potential_label"] == "below_average")]


df = df[df["potential_label"] != "below_average"]



df_pivot_table = pd.pivot_table(df,
                                values="attribute_value",
                                index=["player_id", "position_id", "potential_label"],
                                columns="attribute_id")
df_pivot_table.head()


df_pivot_table = df_pivot_table.reset_index()



label_encoder = LabelEncoder()
df_pivot_table["potential_label"] = label_encoder.fit_transform(df_pivot_table["potential_label"])
df_pivot_table.head()



def grab_col_names(dataframe, cat_th=5, car_th=20):
    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]
    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    return cat_cols, num_cols, cat_but_car


cat_cols, num_cols, cat_but_car = grab_col_names(df_pivot_table)


num_cols = [col for col in num_cols if "player_id" not in str(col) and "position_id" not in str(col)]



df = df_pivot_table
df.head()

scaled = StandardScaler().fit_transform(df[num_cols])

df[num_cols] = pd.DataFrame(scaled, columns=df[num_cols].columns)
df.head()



y = df["potential_label"]
X = df.drop(["potential_label"], axis=1)
X.columns = X.columns.astype(str)

scoring_metrics = ["roc_auc", "f1", "precision", "recall", "accuracy"]

def base_models(X, y, cv=3):
    print("Base Models")
    classifiers = [
        ('LR', LogisticRegression()),
        ("CART", DecisionTreeClassifier()),
        ("RF", RandomForestClassifier()),
        ('Adaboost', AdaBoostClassifier()),
        ('GBM', GradientBoostingClassifier()),
        ('XGBoost', XGBClassifier(eval_metric='logloss')),
        ('LightGBM', LGBMClassifier(verbose=-1)),
    ]

    scoring_metrics = ["roc_auc", "f1", "precision", "recall", "accuracy"]

    for name, classifier in classifiers:
        print(f"{name}:")
        for scoring in scoring_metrics:
            cv_results = cross_val_score(classifier, X, y, cv=cv, scoring=scoring)
            print(f"{scoring}: {round(cv_results.mean(), 4)}")
        print("=" * 50)

base_models(X, y)



cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300, 400]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 1]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

classifiers = [("CART", DecisionTreeClassifier(), cart_params),
               ("RF", RandomForestClassifier(), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
               ('LightGBM', LGBMClassifier(), lightgbm_params)]




def hyperparameter_optimization(X, y, cv=3, scoring="roc_auc"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_).fit(X,y)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model

    return best_models, final_model


best_models, final_model= hyperparameter_optimization(X, y)




def voting_classifier(best_models, X, y):
    print("Voting Classifier...")

    voting_clf = VotingClassifier(estimators=[('RF', best_models["RF"]),
                                              ('CART', best_models["CART"]),
                                              ('XGBoost', best_models["XGBoost"]),
                                              ('LightGBM', best_models["LightGBM"])],
                                  voting='soft').fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)



classifiers = [#('KNN', KNeighborsClassifier(), knn_params),
               ("CART", DecisionTreeClassifier(random_state=222), cart_params),
               ("RF", RandomForestClassifier(random_state=222), rf_params),
               ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=222), xgboost_params),
               ('LightGBM', LGBMClassifier(random_state=222), lightgbm_params)
               ]

def plot_importance(model, features, num=len(X), save=False):
    feature_imp = pd.DataFrame({'Value': model.feature_importances_, 'Feature': features.columns})
    plt.figure(figsize=(10, 10))
    sns.set(font_scale=1)
    sns.barplot(x="Value", y="Feature", data=feature_imp.sort_values(by="Value",
                                                                     ascending=False)[0:num])
    plt.title('Features')
    plt.tight_layout()
    plt.show()
    if save:
        plt.savefig('importances.png')

for name,classifier, params in classifiers:
    gs_best = GridSearchCV(classifier, params, cv=3, n_jobs=-1, verbose=False).fit(X, y)
    final_model = classifier.set_params(**gs_best.best_params_).fit(X, y)
    plot_importance(final_model, X)

plot_importance(final_model,X)

best_models_list = list(best_models.values())

for name, model in best_models_list:
    if name != DecisionTreeClassifier(max_depth=1):
        plot_importance(model, X)
