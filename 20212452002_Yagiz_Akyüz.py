#20212452002 - Yağız AKYÜZ
#20212452004 - Alper BOYACI

#Bu .py dosyası F1 genel istatistiklerini ele alan bir projedir.

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler, MinMaxScaler
import missingno as msno

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 500)

#region Veri Analizi ve Gruplama
df = pd.read_csv("F1DataSet/F12022Results.csv")
df.head()
df.info()
df.shape
df.isnull().values.any()
df.isnull().sum()

#Hangi yarışçı kaç yarışa çıktı?
df["Driver"].value_counts()

#Puan ortalaması
df["Points"].mean()

#Ortalama yarış başlangıç pozisyonu
df["Starting Grid"].mean()

#Yarışçıların numaralarının ortalaması
df["No"].mean()

#Yarışçıların puan ortalaması
df.groupby("Driver").agg({"Points":"mean"})

#Takımların puan ortalaması, toplam puanları ve puan topladıkları yarış sayısı
df.groupby("Team").agg({"Points":["mean", "sum", "count"]})

#Pilotun puan ve tur ortalaması
df.groupby(["Driver", "Team"]).agg({"Points":"mean", "Laps":"mean"})

#Detaylar
df.describe().T
#endregion

#region Yıllara göre pistlerde düzenlenen Grand Prix(GP) sayıları.
races_df = pd.read_csv("F1DataSet/races.csv")
races_df.head()

plt.figure(figsize=(25, 15))
years = np.arange(1950, 2024, 2)
plt.scatter(races_df.year, races_df.name, alpha=0.7, color="b", label = "Yarış")
plt.xlabel("Yıl")
plt.ylabel("Grand Prix")
plt.legend()
plt.grid(True)
plt.xticks(years)
plt.show()
#endregion

#region Pilotların uyrukları
df_f1driver_ds = pd.read_csv("F1DataSet/F1DriversDataset.csv")
df_f1driver_ds.head()

#Boş değer olanları bulup dolduruyoruz
df_f1driver_ds.isnull().sum()
df_f1driver_ds.fillna(0)

#Pilotları uyruklarına göre topluyoruz
driver_count = df_f1driver_ds.groupby('Nationality')['Driver'].count()

df_driver_count = pd.DataFrame(driver_count)
df_driver_count = df_driver_count.sort_values(by='Driver', ascending=False).reset_index()

plt.figure(figsize=(15, 8))
sns.barplot(x='Driver', y='Nationality', data=df_driver_count.head(20), color='blue')
plt.xticks(rotation=90)
plt.show()
#endregion

#region Ülkelerin şampiyonluk sayılar
df_f1driver_ds = pd.read_csv("F1DataSet/F1DriversDataset.csv")

#Ülkeleri ve şampiyonları gruplayıp topluyoruz.
driver_champ = df_f1driver_ds.groupby('Nationality')['Championships'].sum()

df_driver_champ = pd.DataFrame(driver_champ)
df_driver_champ = df_driver_champ.sort_values('Championships', ascending=False).reset_index()
df_driver_champ['Championships'] = df_driver_champ['Championships'].astype('int64')
df_driver_champ.head(10)

plt.figure(figsize = (12,6))
sns.barplot(x='Nationality', y='Championships', data=df_driver_champ.head(10), color='red')
plt.xticks(rotation=90)
plt.show()
#endregion

#region En çok start alan pilotlar
df_f1driver_ds = pd.read_csv("F1DataSet/F1DriversDataset.csv")

#Pilotların aldığı start sayısını topluyoruz
df_race_starters = pd.DataFrame(df_f1driver_ds.groupby('Driver')['Race_Starts'].sum().sort_values(ascending=False)).reset_index()

plt.figure(figsize = (12,6))
sns.barplot(x='Driver', y='Race_Starts', data=df_race_starters.head(10), color='red')

plt.xticks(rotation=90)
plt.show()
#endregion

#region 2022 Verstappen vs Leclerc
results22 = pd.read_csv("F1DataSet/F12022Results.csv")
results22.head()

#bazı column isimlerinin aralarında boşluk olduğu (örn: Starting Grid) ve bunun bize sorun olabileceği için
# basit bir for döngüsü ile bu boşluk olan kısımlara "_" işaretini koyduk.
results22.columns = [x.split()[0] + "_" + x.split()[1] if len(x.split())>1 else x for x in results22.columns]

#verstappen ve leclerc'e kendi verilerini yüklüyoruz
ver = results22.Driver == "Max Verstappen"
lec = results22.Driver == "Charles Leclerc"
verstappen = results22[ver]
leclerc = results22[lec]

print(verstappen.head(3))
print(leclerc.head(3))

races = np.arange(0, 24)
points = np.arange(0, 27)

plt.subplots(figsize=(18, 18))
plt.subplot(2,1,1)

#Yarış ve sıralama sonuclarını grafik olarak hazırlıyoruz
plt.plot(verstappen.Track, verstappen.Points, '-o', color="blue", label="VER Yarış", linewidth=2)
plt.plot(leclerc.Track, leclerc.Points, '-o', color="red", label="LEC Yarış", linewidth=2)

plt.plot(verstappen.Track, verstappen.Starting_Grid, 'b:', color="blue", label="VER Sıralama", linewidth=2)
plt.plot(leclerc.Track, leclerc.Starting_Grid, 'b:', color="red", label="LEC Sıralama", linewidth=2)

plt.grid(True)
plt.xlabel("Yarışlar")
plt.ylabel("Puan ve Sıralama Turları Sonucu")
plt.title("Max VERSTAPPEN vs Charles LECLERC 2022 Sezonu")
plt.xticks(races)
plt.xticks(rotation=90)
plt.yticks(points)
plt.legend()
plt.show()
#endregion

#region Grid pozisyonuna göre alınan puan
results22 = pd.read_csv("F1DataSet/F12022Results.csv")
results22.info()
results22.head()

results22.corr

#Daha düzgün gözüksün diye "No" alanını kaldırdık.
results22.drop(["No"], axis=1, inplace=True)

#Grid'de 1 ile 20 arasında araç olduğu için 1, 21 yaptık. en az 0 maksimum Hızlı tur ile alınabilecek puan 26 olduğu için 0-27 arası yaptık.
grid = np.arange(1,21)
points = np.arange(0,27)

results22.plot(kind="scatter", x="Starting Grid", y="Points", grid=True, color="red", lw=3, alpha=0.4)
plt.xlabel("Grid Pozisyonu")
plt.ylabel("Alınan Puan")
plt.xticks(grid)
plt.yticks(points)
plt.title("Grid Pozisyonuna Göre Alınan Puanlar")
plt.show()
#endregion

#region Turlarda Aykırı Değerler
results21 = pd.read_csv("F1DataSet/F12021Results.csv")

sns.boxplot(x=results21["Laps"])

q1 = results21["Laps"].quantile(0.25)
q3 = results21["Laps"].quantile(0.75)
iqr = q3-q1
low = q1-1.5*iqr
up = q3+1.5*iqr

results21[(results21["Laps"]<low) | (results21["Laps"]>up)].head()
results21[(results21["Laps"]<low) | (results21["Laps"]>up)].index

sonuc = results21[(results21["Laps"]<low) | (results21["Laps"]>up)].any(axis=None)

def outlier_threshold(dataframe, col_name, q1=0.25, q3=0.75):
    q1 = dataframe[col_name].quantile(q1)
    q3 = dataframe[col_name].quantile(q3)
    iqr = q3 - q1
    low_limit = q1 - 1.5 * iqr
    up_limit = q3 + 1.5 * iqr
    return low_limit, up_limit
outlier_threshold(results21, "Laps")
low_limit, up_limit = outlier_threshold(results21, "Laps")

results21[(results21["Laps"]<low_limit) | (results21["Laps"]>up_limit)]
results21[(results21["Laps"]<low_limit) | (results21["Laps"]>up_limit)].index

#region Aykırı Değer Kontrol
def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    result = dataframe[(dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit)].any(axis=None)
    if result:
        return True
    else:
        return False
check_outlier(results21, "Laps")
#endregion

#region Aykırı Değer Sütünları
def grap_col_names(dataframe, cat_th = 10, car_th = 20):
    """
    :parameter
    -------------
    dataframe : dataframe
        değişken isimleri alınmak istenen dataframe'dir.
    cat_th : int, float
        numerik olan fakat kategorik olan değişkenler için sınıf eşik değeridirç
    car_th : int, float
        kategorik fakat kardinal değişkenler için sınıf eşik değeri.

    :returns

    cat_cols : list
        Kategorik değişken listesi
    num_cols : list
        Numerik değişken listesi
    cat_but_car : list
        Kategorik görünümlü kardinal değişken listesi

    notes:
    Ek bilgi içermemektedir.
    """
    cat_cols = [col for col in dataframe.columns if str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < 10 and dataframe[col].dtypes in ["int", "float"]]

    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > 20 and str(dataframe[col].dtypes) in ["category", "object", "bool"]]

    cat_cols = cat_cols + num_but_cat

    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes in ["int", "float"]]
    num_col = [col for col in num_cols if col not in cat_cols]

    print("Gözlem Sayısı", dataframe.shape[0])
    print("Değişken Sayısı", dataframe.shape[1])
    print("Kategorik Sayısı", len(cat_cols))
    print("Numerik Değişken Sayısı", len(num_cols))
    print("Kardinal Olan Kategorik Değişkenler", len(cat_but_car))
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grap_col_names(results21)
for col in num_cols:
    print(col, check_outlier(results21, col))

def grab_outliers(dataframe, col_name, index=False):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    aykiri = ((dataframe[col_name]<low_limit) | (dataframe[col_name]>up_limit))
    if aykiri.shape[0]>10:
        print(dataframe[aykiri].head())
    else:
        print(dataframe[aykiri])
    if index:
        aykiri_index = dataframe[aykiri].index
        return aykiri_index
grab_outliers(results21, "Laps", index=True)
#endregion

#region Aykırı Değer Silme
def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_threshold(dataframe, col_name)
    yepis_df = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return yepis_df
dff = remove_outlier(results21, "Laps")
check_outlier(dff, "Laps")
results21["Laps"][19]
#endregion

#endregion

#region Label Encoding
df = pd.read_csv("F1DataSet/F12021Results.csv")

# "+1 Pt" sütununu içinde ki 0-1 değer toplamını kontrol ediyoruz
df["+1 Pt"].value_counts()

#1-0 çevirmesi yapıyoruz
def label_encoder(dataframe, binary_col):
    le = LabelEncoder()
    dataframe[binary_col] = le.fit_transform(df[binary_col])
    return dataframe
label_encoder(df, "+1 Pt")

binary_cols = [col for col in df.columns if df[col].dtype not in ["int", "float"] and df[col].nunique()]

#Başka yapılabilecek kolonlarıda yapıyoruz
for col in binary_cols:
    label_encoder(df, col)

df.head()
#endregion

#region ONE HOT ENCODER
df = pd.read_csv("F1DataSet/F12021Results.csv")
df["Position"].value_counts()
df["+1 Pt"].value_counts()
ohe_cols = [col for col in df.columns if df[col].nunique()>2 and df[col].nunique()<10]

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first).head()
    return dataframe

one_hot_encoder(df, ["+1 Pt"])
one_hot_encoder(df, ["Position"])
#endregion

#region Kayıp Veriler ve Temizlenmesi
results21 = pd.read_csv("F1DataSet/F12021Results.csv")

def missing_values_table(dataframe, na_col=False):
    na_cols = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_cols].isnull().sum().sort_values(ascending=True)
    ratio = (dataframe[na_cols].isnull().sum() / dataframe[na_cols].shape[0]*100).sort_values(ascending=True)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=["n_miss", "ratio"])
    print(missing_df, end="\n")
    if na_col:
        return na_cols
missing_values_table(results21)

results21["Fastest Lap"].fillna("NFL")

results21 = results21.dropna()

msno.bar(results21)
msno.matrix(results21)
#endregion

#region Feaure Scaling - Standartlaştırma

#Klasik Standartlaştırma - Mean()
df = pd.read_csv("F1DataSet/F12021Results.csv")

ss = StandardScaler()
df["laps_standart_scaler"] = ss.fit_transform(df[["Laps"]])
df.head()

#Robust Yönetimi - Median()
rs = RobustScaler()
df["laps_robust_scaler"] = rs.fit_transform(df[["Laps"]])
df.head()

#MinMax Scaler - Min Max değeri arasında oluşturuyor
mms = MinMaxScaler()
df["laps_min_max_scaler"] = mms.fit_transform(df[["Laps"]])
df.head()
#endregion