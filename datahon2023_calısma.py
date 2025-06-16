import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
#from ydata_profilling import ProfileReport 

#%% veriyi anlama kısmı (EDA)

df = pd.read_csv("train.csv")

df.isnull().sum()


def basic_info(df):
    print("şekil: ",  df.shape)
    print("ilk 5 satır:\n", df.head())
    print("boş değerler:\n", df.isnull().sum())
    print("veri tipleri:\n", df.dtypes)
    

basic_info(df)


def categorical_summary(df):
    df_cat = df.select_dtypes(include = "object").columns
    
    for col in df_cat:
        print(f"\n {col} frekans dağılımı:\n{df[col].value_counts()}")
        
categorical_summary(df)


def numerical_summary(df):
    num_col = df.select_dtypes(exclude = "object").columns
    
    print("\n Sayısal özet:\n", df[num_col].describe())

numerical_summary(df)



#%% Preprocessing

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split



def encode_features(df):
    df = df.copy()
    
    le = LabelEncoder()
    df["Öbek İsmi"] = le.fit_transform(df["Öbek İsmi"])
    
    # cat için one hot yapalım
    one_hot_columns = ['Cinsiyet', 'Yaş Grubu', 'Medeni Durum', 'Eğitim Düzeyi',
           'İstihdam Durumu', 'Yaşadığı Şehir', 'En Çok İlgilendiği Ürün Grubu',
           'Eğitime Devam Etme Durumu']
    df = pd.get_dummies(df, columns = one_hot_columns, dtype=int)
    
    return df, le




def scale_numeric_features(df, scaler = None):
    df = df.copy()
    
    num_cols = ['Yıllık Ortalama Gelir', 'Yıllık Ortalama Satın Alım Miktarı',
           'Yıllık Ortalama Sipariş Verilen Ürün Adedi',
           'Yıllık Ortalama Sepete Atılan Ürün Adedi']
    
    if scaler is None:
        scaler = StandardScaler()
        
    df[num_cols] = scaler.fit_transform(df[num_cols])
        
    return df, scaler

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






































































































