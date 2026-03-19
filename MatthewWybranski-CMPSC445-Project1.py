# MatthewWybranski-CMPSC445-Project1

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

# 1. Data Collection

temp_anom_df = pd.read_csv("GLB.Ts+dSST.csv", skiprows=1)
ch4_df = pd.read_csv("ch4_annmean_gl.csv", skiprows=43)
co2_df = pd.read_csv("co2_annmean_mlo.csv", skiprows=43)
n2o_df = pd.read_csv("n2o_annmean_gl.csv", skiprows=43)
owid_df = pd.read_csv("owid-co2-data.csv")


gases_df = pd.merge(co2_df, ch4_df, how='outer', on='year')
gases_df = pd.merge(gases_df, n2o_df, how='outer', on='year')
gases_df.columns = ['year', 'co2_mean', 'co2_unc',
                    'ch4_mean', 'ch4_unc', 'n2o_mean', 'n2o_unc']
gases_df = gases_df.drop(columns=['co2_unc', 'ch4_unc', 'n2o_unc'])

owid_df_pruned = owid_df[owid_df['country'] == 'World']
owid_df_pruned = owid_df_pruned.drop(columns=['country', 'iso_code'])

inputs_df = pd.merge(gases_df, owid_df_pruned, how='outer', on='year')

temp_anom_df_pruned = temp_anom_df.drop(columns=[
    'Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct',
    'Nov','Dec','D-N','DJF','MAM','JJA','SON'])

temp_anom_df_pruned.rename(columns={'Year':'year'}, inplace=True)

full_df = pd.merge(inputs_df, temp_anom_df_pruned, how='outer', on='year')

df = full_df.sort_values("year").reset_index(drop=True)

print(df.shape)

# 2. Data Preprocessing

df["years_since_start"] = df["year"] - df["year"].min()

df["co2_growth"] = df["co2_mean"].diff()
df["ch4_growth"] = df["ch4_mean"].diff()
df["n2o_growth"] = df["n2o_mean"].diff()

df["J-D"] = pd.to_numeric(df["J-D"], errors='coerce')
df["co2_ma_5"] = df["co2_mean"].rolling(5).mean()
df["temp_ma_5"] = df["J-D"].rolling(5).mean()

df["co2_mean"] = df["co2_mean"].interpolate()
df["ch4_mean"] = df["ch4_mean"].interpolate()
df["n2o_mean"] = df["n2o_mean"].interpolate()

df = df.ffill().bfill()

print(df[0:5])

# 3. Model Development and 4. Root-Cause Identification via Feature Ranking

X = df.drop(columns=["year", "J-D"])
y = df["J-D"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=360, shuffle=False
)

# feature selector created
fs = SelectKBest(score_func=f_regression, k=10)

# fit using training data
X_train_selected = fs.fit_transform(X_train, y_train)

# transform test data
X_test_selected = fs.transform(X_test)

print(X_train_selected.shape)
print("Selected features:")
print(X.columns[fs.get_support()])

lr = LinearRegression()
lr.fit(X_train_selected, y_train)

y_pred = lr.predict(X_train_selected)

print("MSE:", mean_squared_error(y_train, y_pred))
print("R2:", r2_score(y_train, y_pred))

selected_features = X.columns[fs.get_support()]  # names of the 10 features selected
coefficients = pd.Series(lr.coef_, index=selected_features)
coefficients = coefficients.reindex(coefficients.abs().sort_values(ascending=False).index)
print(coefficients)

# 5. Visualization Requirements
# Feature importance bar charts
plt.figure()
coefficients.head(10).plot(kind="bar")
plt.title("Feature Importance (Linear Regression)")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

# Time‑series trends of greenhouse gases vs temperature
plt.figure()

fig, ax1 = plt.subplots()

ax1.set_xlabel("Year")
ax1.set_ylabel("Temperature Anomaly")
ax1.plot(df["year"], df["J-D"], label="Temperature")

ax2 = ax1.twinx()
ax2.set_ylabel("GHG per capita")
ax2.plot(df["year"], df["ghg_per_capita"], linestyle="--", label="GHG")

plt.title("Temperature vs GHG (Dual Axis)")
fig.tight_layout()
plt.legend()
plt.show()

# Time‑series trends of CO2 vs temperature
plt.figure()

fig, ax1 = plt.subplots()

ax1.set_xlabel("Year")
ax1.set_ylabel("Temperature Anomaly")
ax1.plot(df["year"], df["J-D"], label="Temperature")

ax2 = ax1.twinx()
ax2.set_ylabel("CO2 (ppm)")
ax2.plot(df["year"], df["co2_mean"], linestyle="--", label="CO2")

plt.title("Temperature vs CO2 (Dual Axis)")
fig.tight_layout()
plt.legend()
plt.show()

# CO2 mean vs Temp
plt.figure()
plt.scatter(df["co2_mean"], df["J-D"])
plt.xlabel("CO2")
plt.ylabel("Temperature")
plt.title("CO2 vs Temperature")
plt.show()

# Take the top 3 features by standardized coefficient magnitude
top_features = coefficients.head(3).index

plt.figure(figsize=(15, 4))

for i, feature in enumerate(top_features):
    plt.subplot(1, 3, i+1)
    plt.scatter(df[feature], df['J-D'], alpha=0.7)
    plt.xlabel(feature)
    plt.ylabel('Temperature Anomaly (J-D)')
    plt.title(f'{feature} vs Temperature')

plt.tight_layout()
plt.show()