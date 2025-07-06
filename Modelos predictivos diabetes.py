import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score
from xgboost import XGBClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import roc_curve, roc_auc_score


#lectura del dataset
df_total = pd.read_csv(r"C:\Users\34601\Desktop\Trabajos en curso\TFM\Dataset\diabetes_binary.csv")

#limpieza de datos nulos
df_total = df_total.dropna()


#print(df_total.shape)

#copia del dataset para los escalados
df_escal = df_total.copy()

#print(df_total.head())

correlation_matrix = df_total.corr()

# Visualización mapa de correlación
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de correlación entre variables")
plt.show()


# Filtrar solo correlación con la variable objetivo
target_corr = correlation_matrix[["Diabetes_binary"]].sort_values(by="Diabetes_binary", ascending=False)

# Visualizar heatmap
plt.figure(figsize=(6, 12))
sns.heatmap(target_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlación de las variables con Diabetes_binary")
plt.show()

sns.boxplot(x=df_total['BMI'])
plt.title('Boxplot de BMI')
plt.show()

# Eliminar filas con valores atípicos en BMI
df_total = df_total[df_total['BMI'] <= 60]


# eliminar columas con baja correlación con la variable objetivo
columnas_baja_correlacion = ['Education', 'PhysActivity','HvyAlcoholConsump','Veggies','Fruits','MentHlth',
                             'CholCheck','Stroke','Smoker','Sex', 'NoDocbcCost','AnyHealthcare',]
df_total = df_total.drop(columns=columnas_baja_correlacion)



# One-hot encoding para variables ordinales
categoricas_ordinales = ['Income']
df_escal = pd.get_dummies(df_escal, columns=categoricas_ordinales, prefix=categoricas_ordinales)
# Escalar variables continuas
columnas_continuas = ['BMI','PhysHlth','GenHlth']
scaler = StandardScaler()
df_escal[columnas_continuas] = scaler.fit_transform(df_escal[columnas_continuas])



#dividir entrenamiento
X = df_total.drop(columns=['Diabetes_binary'])
y = df_total['Diabetes_binary']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# dividir entrenamiento con escalado
X_escal = df_escal.drop(columns=['Diabetes_binary'])
y_escal = df_escal['Diabetes_binary']
X_train_escal, X_test_escal, y_train_escal, y_test_escal = train_test_split(X_escal, y_escal, test_size=0.2, random_state=42, stratify=y_escal)




#  Regresión logistica
model_lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model_lr.fit(X_train_escal, y_train_escal)
y_probs = model_lr.predict_proba(X_test_escal)[:, 1]
threshold = 0.35
y_pred = (y_probs >= threshold).astype(int)

print("Regresión Logística")
print("Sensibilidad:", f"{recall_score(y_test_escal, y_pred)*100:.2f}%")
print("Exactitud:",f"{accuracy_score(y_test_escal, y_pred)*100:.2f}%")
print("F1 scorer:", f"{f1_score(y_test_escal, y_pred)*100:.2f}%")
print("Matriz de confusión:\n", confusion_matrix(y_test_escal, y_pred))

fpr, tpr, thresholds = roc_curve(y_test_escal, y_probs)
auc_score = roc_auc_score(y_test_escal, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray') 
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC - Regresión Logística')
plt.legend(loc='lower right')
plt.grid()
plt.show()

# Random forest

model_rf = RandomForestClassifier(n_estimators=200,class_weight='balanced', random_state=42)
model_rf.fit(X_train, y_train)
y_probs = model_rf.predict_proba(X_test)[:, 1]
threshold = 0.35
y_pred = (y_probs >= threshold).astype(int)

print("Random Forest")
print("Sensibilidad:", f"{recall_score(y_test, y_pred)*100:.2f}%")
print("Exactitud:",f"{accuracy_score(y_test, y_pred)*100:.2f}%")
print("F1 scorer:", f"{f1_score(y_test, y_pred)*100:.2f}%")
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray') 
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC - Random Forest')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# XGBOOST
model_xgb = XGBClassifier(eval_metric='logloss', random_state=42)
model_xgb.fit(X_train, y_train)
y_probs = model_xgb.predict_proba(X_test)[:, 1]
threshold = 0.35
y_pred = (y_probs >= threshold).astype(int)  

print("XGBoost")
print("Sensibilidad:", f"{recall_score(y_test, y_pred)*100:.2f}%")
print("Exactitud:",f"{accuracy_score(y_test, y_pred)*100:.2f}%")
print("F1 scorer:", f"{f1_score(y_test, y_pred)*100:.2f}%")
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray') 
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC - XGBoost')
plt.legend(loc='lower right')
plt.grid()
plt.show()


# KNN 
for k in [3, 9, 15, 21]:
   
    model_knn = KNeighborsClassifier(n_neighbors=k, weights='distance')
    model_knn.fit(X_train_escal, y_train_escal)
    y_probs = model_knn.predict_proba(X_test_escal)[:, 1]
    threshold = 0.35
    y_pred = (y_probs >= threshold).astype(int)



    print(f"KNN con k={k}")
    print("Sensibilidad:", f"{recall_score(y_test_escal, y_pred)*100:.2f}%")
    print("Exactitud:",f"{accuracy_score(y_test_escal, y_pred)*100:.2f}%")
    print("F1 scorer:", f"{f1_score(y_test_escal, y_pred)*100:.2f}%")
    print("Matriz de confusión:\n", confusion_matrix(y_test_escal, y_pred))

    fpr, tpr, thresholds = roc_curve(y_test_escal, y_probs)
    auc_score = roc_auc_score(y_test_escal, y_probs)

    plt.figure()
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray') 
    plt.xlabel('Tasa de falsos positivos (FPR)')
    plt.ylabel('Tasa de verdaderos positivos (TPR)')
    plt.title('Curva ROC - KNN')
    plt.legend(loc='lower right')
    plt.grid()
    plt.show()

# Naive Bayes
model_nb = BernoulliNB()
model_nb.fit(X_train, y_train)
y_probs = model_nb.predict_proba(X_test)[:, 1]
threshold = 0.3
y_pred = (y_probs >= threshold).astype(int)

print("Naive Bayes")
print("Sensibilidad:", f"{recall_score(y_test, y_pred)*100:.2f}%")
print("Exactitud:",f"{accuracy_score(y_test, y_pred)*100:.2f}%")
print("F1 scorer:", f"{f1_score(y_test, y_pred)*100:.2f}%")
print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray') 
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC - Naive Bayes')
plt.legend(loc='lower right')
plt.grid()
plt.show()

