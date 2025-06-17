import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lofo import LOFOImportance, Dataset, plot_importance
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier
from tqdm.autonotebook import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
#%% veriyi anlama kısmı (EDA)

df = pd.read_csv("train.csv")


def basic_info(df):
    
    print("\n ilk 5 satır: " , df.head())
    print("\n shapine bakalım: ", df.shape)
    print("\n eksik değer var mı: ", df.isnull().sum())
    print("\n veri tiplerine bakalım ", df.dtypes)
    

def categorical_summary(df):
    
    cat_col = df.select_dtypes(include = "object").columns
    
    for col in cat_col:
        print(f"\n {col} için value countslar: \n\n", df[col].value_counts())
        

def numerical_summary(df):
    
    num_col = df.select_dtypes(exclude = "object").columns
    
    for col in num_col:
        print(f"\n\n\n{col} için describe değerleri: \n\n\n", df[col].describe())
    
def correlation_matrix(df):
    
    df_corr = df.select_dtypes(exclude = "object").corr()
    sns.heatmap(df_corr, annot = True)
    plt.show()

#%% Preprocessing

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split


def LabelEn_and_OneHot(df):
    
    df = df.copy()
    
    le = LabelEncoder()
    
    df["Öbek İsmi"] = le.fit_transform(df["Öbek İsmi"])
    
    one_hot_columns = ['Cinsiyet', 'Yaş Grubu', 'Medeni Durum', 'Eğitim Düzeyi',
           'İstihdam Durumu', 'Yaşadığı Şehir', 'En Çok İlgilendiği Ürün Grubu',
           'Eğitime Devam Etme Durumu']
    
    df = pd.get_dummies(df, columns = one_hot_columns, dtype=int)
    
    return df, le
    

def standarScaler(df):
    
    df = df.copy()
    scaler = StandardScaler()
    
    num_columns = ['Yıllık Ortalama Gelir', 'Yıllık Ortalama Satın Alım Miktarı',
           'Yıllık Ortalama Sipariş Verilen Ürün Adedi',
           'Yıllık Ortalama Sepete Atılan Ürün Adedi']
    
    df[num_columns] = scaler.fit_transform(df[num_columns])
    
    return df, scaler



def lofo_feature_importance_classification(df, target_col="Öbek İsmi", model=None, cv_splits=5, n_jobs=-1, plot=True):
    

    X = df.drop(columns=[target_col, "index"])
    y = df[target_col]

    dataset = Dataset(df=df, target=target_col, features=X.columns.tolist())
    
    # Stratified çünkü classification problemi
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    lofo = LOFOImportance(dataset, model=model, cv=cv, scoring="accuracy", n_jobs=n_jobs)
    importance_df = lofo.get_importance()

    if plot:
        plot_importance(importance_df, figsize=(12, 20))
        plt.show()
    
    return importance_df


def split_data(df):
    
    X = df.drop(["Öbek İsmi", "index"], axis = 1)
    y = df["Öbek İsmi"]
    
    return train_test_split(X, y, test_size = 0.20, random_state = 42, shuffle = True)



#%% modeli train edelim

def train_and_tune_models(X_train, X_test, y_train, y_test):
    models_and_params = {
        "LogisticRegression": (
            LogisticRegression(max_iter=1000),
            {"C": [0.01, 0.1, 1, 10], "solver": ["liblinear", "lbfgs"]}
        ),
        "DecisionTree": (
            DecisionTreeClassifier(),
            {"max_depth": [3, 5, 10, None], "criterion": ["gini", "entropy"]}
        ),
        "RandomForest": (
            RandomForestClassifier(),
            {"n_estimators": [100, 200], "max_depth": [None, 10, 20]}
        ),
        "SVC": (
            SVC(),
            {"C": [0.1, 1, 10], "kernel": ["linear", "rbf"]}
        ),
        "XGBoost": (
            XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
            {"n_estimators": [100, 200], "max_depth": [3, 5], "learning_rate": [0.01, 0.1]}
        )
    }

    best_models = {}

    for model_name, (model, params) in models_and_params.items():
        print(f"\n🔍 Model: {model_name}")

        grid = GridSearchCV(model, param_grid=params, cv=5, scoring="accuracy", n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        best_params = grid.best_params_
        cv_score = grid.best_score_

        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        print(f"✅ En iyi parametreler: {best_params}")
        print(f"📈 CV doğruluğu: {cv_score:.4f}")
        print(f"🧪 Test doğruluğu: {acc:.4f}")
        print("\n📊 Classification Report:\n", classification_report(y_test, y_pred))
        print("🧩 Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

        best_models[model_name] = best_model

    return best_models



#%% fonksyonları çağıralım

df_encoded, le = LabelEn_and_OneHot(df)

df_scaler_encoded, scaler = standarScaler(df_encoded)

lofo_result = lofo_feature_importance_classification(
    df=df_scaler_encoded,
    target_col="Öbek İsmi",
    model=LGBMClassifier(),   
    cv_splits=5,
    n_jobs=-1,
    plot=True
)


X_train, X_test, y_train, y_test = split_data(df_scaler_encoded)


best_models = train_and_tune_models(X_train, X_test, y_train, y_test)






















































































































#%% veri setini train ve test olarak ayırma 

def split_data(df, target_col = "Öbek İsmi", test_size = 0.20, random_state = 42):
    X = df.drop([target_col, 'index'], axis = 1)
    y = df[target_col]
    return train_test_split(X,y, test_size= test_size, random_state= random_state, shuffle=True)


#%% fonskyonları çağıralım ve sorun var mı bakalım


df_encoded, label_encoder = encode_features(df)


df_scaled, scaler = scale_numeric_features(df_encoded)


X_train, X_test, y_train, y_test = split_data(df_scaled)


#%% model train ve test edelim 

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report




def train_model(X_train, X_test, y_train, y_test):
    
    model = LogisticRegression(max_iter=1000)
    
    # cross val
    scores = cross_val_score(model, X_train, y_train, cv=5, scoring = "accuracy")
    print("cross-val skorları: ", scores)
    print("ortalama doğruluk: ", scores.mean())
    
    
    # modeli tüm veri ile eğit
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred)
    print("test seti doğruluk: ", test_accuracy)
    print("\n classification report: ", classification_report(y_test, y_pred))
    
    
    
    return model

model = train_model(X_train, X_test, y_train, y_test)






































































































