import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_diabetes

# Caricamento dataset

data = load_diabetes(as_frame=True)

X = data.data
y = data.target

# Creazione DataFrame completo

df = data.frame

# Esplorazione iniziale

print("Dimensioni del dataset:")
print(df.shape)

print("\nPrime 5 righe:")
print(df.head())

print("\nInformazioni sul dataset:")
print(df.info())

print("\nAnalisi descrittiva:")
print(df.describe())

# Istogrammi delle variabili

df.hist(figsize=(15, 12), bins=20)
plt.suptitle("Distribuzione delle variabili - Diabetes")
plt.tight_layout()
plt.show()

# Boxplot delle feature

plt.figure(figsize=(14, 6))
sns.boxplot(data=X)
plt.title("Boxplot delle feature - Diabetes")
plt.xticks(rotation=45)
plt.show()

# Scatter matrix

pd.plotting.scatter_matrix(
    df,
    figsize=(16, 16),
    diagonal="hist",
    alpha=0.5
)

plt.suptitle("Scatter Matrix - Diabetes")
plt.tight_layout()
plt.show()

from sklearn.model_selection import train_test_split

# Separazione delle variabili
X = df.drop("target", axis=1)
y = df["target"]

# Suddivisione Training e Test

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Dimensioni Training:", X_train.shape)
print("Dimensioni Test:", X_test.shape)
print("Dimensioni y_train:", y_train.shape)
print("Dimensioni y_test:", y_test.shape)

# FASE 2 - confronto dei modelli #

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

from sklearn.preprocessing import StandardScaler

# Definizione dei modelli

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(random_state=42),

    "Ridge": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Ridge())
    ]),

    "Lasso": Pipeline([
        ("scaler", StandardScaler()),
        ("model", Lasso(max_iter=10000))
    ]),

    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsRegressor())
    ]),

    "SVR": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVR())
    ])
}

print("Modelli definiti:")
for name in models:
    print(name)

    # K-Fold Cross Validation

kf = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# Confronto dei modelli tramite NMSE

results = {}

for name, model in models.items():

    scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=kf,
        scoring="neg_mean_squared_error"
    )

    results[name] = scores

    print(f"\n{name}")
    print("NMSE per fold:", scores)
    print("NMSE medio:", scores.mean())
    print("MSE medio:", -scores.mean())

# Confronto grafico dei modelli

model_names = list(results.keys())

mean_mse = [
    -scores.mean()
    for scores in results.values()
]

plt.figure(figsize=(10, 6))

plt.bar(model_names, mean_mse)

plt.title("Confronto modelli - K-Fold Cross Validation")
plt.xlabel("Modelli")
plt.ylabel("Mean Squared Error")

plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


# FASE 3 - HYPERPARAMETERS TUNING #


from sklearn.model_selection import GridSearchCV

# Pipeline Ridge

ridge_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", Ridge())
])

# Griglia degli iperparametri

param_grid = {
    "model__alpha": [0.01, 0.1, 1, 10, 100, 1000]
}

# Grid Search

grid_search = GridSearchCV(
    ridge_pipeline,
    param_grid,
    cv=5,
    scoring="neg_mean_squared_error"
)

grid_search.fit(X_train, y_train)

# Miglior modello

best_model = grid_search.best_estimator_

print("Migliori parametri:")
print(grid_search.best_params_)

print("\nMiglior NMSE:")
print(grid_search.best_score_)

# Valutazione finale sul Test Set

from sklearn.metrics import mean_squared_error, r2_score

# Predizioni

y_pred = best_model.predict(X_test)

# Metriche

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("RISULTATI MODELLO OTTIMIZZATO")
print("Miglior Alpha:", grid_search.best_params_["model__alpha"])
print(f"MSE sul Test Set: {mse:.4f}")
print(f"R2 sul Test Set: {r2:.4f}")


# FASE 4 - LEARNING CURVES #

from sklearn.model_selection import learning_curve

train_sizes, train_scores, validation_scores = learning_curve(
    best_model,
    X_train,
    y_train,
    cv=5,
    scoring="neg_mean_squared_error",
    train_sizes=np.linspace(0.1, 1.0, 10),
    shuffle=True,
    random_state=42
)

# Conversione in MSE positivo

train_mse = -train_scores
validation_mse = -validation_scores

# Medie

train_mean = np.mean(train_mse, axis=1)
validation_mean = np.mean(validation_mse, axis=1)

# Grafico

plt.figure(figsize=(10, 6))

plt.plot(train_sizes, train_mean, marker="o", label="Training MSE")
plt.plot(train_sizes, validation_mean, marker="o", label="Validation MSE")

plt.title("Learning Curves - Ridge ottimizzato")
plt.xlabel("Numero di campioni di Training")
plt.ylabel("Mean Squared Error")

plt.legend()
plt.grid()
plt.show()


# FASE 5 - PCA E VISUALIZZAZIONE #

from sklearn.decomposition import PCA

# Applicazione PCA sulle feature di training

pca = PCA(n_components=2)

X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# Varianza spiegata

print("Varianza spiegata PC1 e PC2:")
print(pca.explained_variance_ratio_)

print("\nVarianza cumulativa:")
print(pca.explained_variance_ratio_.sum())

# Ridge sulle due componenti principali

ridge_pca = Ridge(
    alpha=grid_search.best_params_["model__alpha"]
)

ridge_pca.fit(X_train_pca, y_train)

# Predizioni

y_pred_pca = ridge_pca.predict(X_test_pca)

print("R2 del modello Ridge con PCA:",
      r2_score(y_test, y_pred_pca))


# Grafico 3D del piano di regressione

from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection="3d")

# Punti reali

ax.scatter(
    X_test_pca[:, 0],
    X_test_pca[:, 1],
    y_test,
    label="Valori reali"
)

# Griglia per il piano

pc1_range = np.linspace(
    X_test_pca[:, 0].min(),
    X_test_pca[:, 0].max(),
    30
)

pc2_range = np.linspace(
    X_test_pca[:, 1].min(),
    X_test_pca[:, 1].max(),
    30
)

PC1, PC2 = np.meshgrid(pc1_range, pc2_range)

Z = ridge_pca.predict(
    np.c_[PC1.ravel(), PC2.ravel()]
).reshape(PC1.shape)

# Piano di regressione

ax.plot_surface(
    PC1, PC2, Z,
    alpha=0.4
)

ax.set_title("Piano di regressione Ridge nello spazio PCA")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("Target")

ax.legend()
plt.show()