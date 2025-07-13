import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, recall_score, ConfusionMatrixDisplay, class_likelihood_ratios
from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from matplotlib.gridspec import GridSpec
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay


def reclasificar_BMI(data):
    data["BMI_reclass"] = data["BMI"]
    for i in range(len(data["BMI_reclass"])):
        if data.loc[i,'BMI_reclass'] < 18.5:
            data.loc[i,'BMI_reclass'] = 0
        elif 18.5 <= data.loc[i,'BMI_reclass'] < 25:
            data.loc[i,'BMI_reclass'] = 1
        elif 25 <= data.loc[i,'BMI_reclass'] < 30:
            data.loc[i,'BMI_reclass'] = 2
        elif 30 <= data.loc[i,'BMI_reclass'] < 35:
            data.loc[i,'BMI_reclass'] = 3
        elif 35 <= data.loc[i,'BMI_reclass'] < 40:
            data.loc[i,'BMI_reclass'] = 4
        else:
            data.loc[i,'BMI_reclass'] = 5

    return data["BMI_reclass"]

def prob_curve_plots(clf_list, X_train, y_train, X_test, y_test):

    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(4, 2)
    colors = plt.get_cmap("Dark2")

    ax_calibration_curve = fig.add_subplot(gs[:2, :2])
    calibration_displays = {}
    for i, (clf, name) in enumerate(clf_list):
        clf.fit(X_train, y_train)
        display = CalibrationDisplay.from_estimator(
            clf,
            X_test,
            y_test,
            n_bins=10,
            name=name,
            ax=ax_calibration_curve,
            color=colors(i),
        )
        calibration_displays[name] = display

    ax_calibration_curve.grid()
    ax_calibration_curve.set_title("Calibration plots (Naive Bayes)")

    # Add histogram
    grid_positions = [(2, 0), (2, 1), (3, 0), (3, 1)]
    for i, (_, name) in enumerate(clf_list):
        row, col = grid_positions[i]
        ax = fig.add_subplot(gs[row, col])

        ax.hist(
            calibration_displays[name].y_prob,
            range=(0, 1),
            bins=10,
            label=name,
            color=colors(i),
        )
        ax.set(title=name, xlabel="Mean predicted probability", ylabel="Count")

    plt.tight_layout()
    plt.show()

def tunning_rf_model(X_train, y_train, thresh, optimization):
    
    model = RandomForestClassifier(random_state=42, n_jobs=-1)

    if optimization == 1:
    
        param_grid = {
        'n_estimators': [30,100],  # Adjust the number of trees in the forest
        'max_depth': [10],  # Adjust the maximum depth of each tree
        'min_samples_split': [10],  # Adjust the minimum samples required to split a node
        'min_samples_leaf': [4]  # Adjust the minimum samples required in a leaf node
    }

        grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        best_model.fit(X_train, y_train)

        train_accuracy = best_model.score(X_train, y_train)
        val_accuracy = best_model.score(X_test, y_test)

        # y_pred = best_model.predict(X_test)
        
        y_probs = best_model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= thresh).astype(int)

        # Print the results
        print("Random Forest")
        print("Best Hyperparameters:", grid_search.best_params_)
        print("Best Cross-Validation Score:", grid_search.best_score_) 
        print("Check Parameters:", best_model.get_params())
    
    elif optimization == 0: 

        model.fit(X_train, y_train)
        y_probs = model.predict_proba(X_test)[:, 1]
        y_pred = (y_probs >= thresh).astype(int)

        
        print("Random Forest")
        print("Parameters:", model.get_params())

    
    print("Results:", classification_report(y_pred, y_test, target_names=['No Diabetes', 'Diabetes']))
    print("DOR:",class_likelihood_ratios(y_test, y_pred, labels=[0, 1]))
    cm = confusion_matrix(y_test, y_pred)
    cm_display = ConfusionMatrixDisplay(cm).plot()
    plt.savefig('RF_conf_mat.png',dpi=300)

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
    plt.savefig('RF_ROC_curve.png',dpi=300)

#Configuración
escalado = 0
optimization = 1
label_column = "Diabetes_binary"
clf_list = ['LR', 'RF', 'XGB', 'KNN', 'NB']
threshold = 0.3

#lectura del dataset
df_total = pd.read_csv(r"c:\Users\34601\Desktop\TFM\Dataset\diabetes_binary.csv")
#limpieza de datos nulos
df_total = df_total.dropna()

df_total_total = df_total.copy()
#print(df_total.shape)

#copia del dataset para los escalados
df_escal = df_total.copy()

# Eliminar filas con valores atípicos en BMI
df_total = df_total[df_total['BMI'] <= 60]
df_total_total['BMI_reclass'] = reclasificar_BMI(df_total_total)
df_total = df_total_total
df_total = df_total.drop(columns=['BMI'])

# sns.pairplot(data = df_total, hue = label_column )
# plt.show()

# Correlacion de las variables con la variable objetivo
correlation_matrix = df_total.corr()

# Visualización mapa de correlación
plt.figure(figsize=(14, 10))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matriz de correlación entre variables")
plt.show()

## Ordenar nivel correlación con la variable objetivo
target_corr = correlation_matrix[[label_column]].sort_values(by=label_column, ascending=False)
plt.figure(figsize=(6, 12))
sns.heatmap(target_corr, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.title("Correlación de las variables con" + label_column)
plt.show()

## Eliminar columnas con baja correlación
columnas_baja_correlacion = ['Education', 'PhysActivity','HvyAlcoholConsump','Veggies','Fruits','MentHlth',
                             'CholCheck','Stroke','Smoker','Sex', 'NoDocbcCost','AnyHealthcare']
# df_total = df_total.drop(columns=columnas_baja_correlacion)

#dividir entrenamiento con o sin escalado
if escalado == 0:
    X = df_total.drop(columns=[label_column])
    y = df_total[label_column]
    
elif escalado == 1:
    columnas_continuas = ['BMI','PhysHlth','GenHlth']
    scaler = StandardScaler()
    df_escal[columnas_continuas] = scaler.fit_transform(df_escal[columnas_continuas])

    X_escal = df_escal.drop(columns=[label_column])
    y_escal = df_escal[label_column]
    X = X_escal
    y = y_escal
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

fig,axs = plt.subplots(2,1)
axs[0].pie(y_train.value_counts(),labels=y_train.unique(),autopct = '%1.2f%%',shadow = True)
axs[0].set_title('Training Dataset')

axs[1].pie(y_test.value_counts(),labels=y_test.unique(),autopct = '%1.2f%%', shadow =True)
axs[1].set_title('Test Dataset')

plt.tight_layout()

plt.show()

#  Regresión logistica
model_lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
model_lr.fit(X_train, y_train)
y_probs = model_lr.predict_proba(X_test)[:, 1]
y_pred = (y_probs >= threshold).astype(int)

print("Regresión Logística")
print("Sensibilidad:", f"{recall_score(y_test, y_pred)*100:.2f}%")
print("Exactitud:",f"{accuracy_score(y_test, y_pred)*100:.2f}%")
print("F1 scorer:", f"{f1_score(y_test, y_pred)*100:.2f}%")
print("DOR:",class_likelihood_ratios(y_test, y_pred, labels=[0, 1]))
# print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
cm_display = ConfusionMatrixDisplay(cm).plot()
cm_display.ax_.set_title('Matriz de confusión-Regresión Logística')
plt.savefig('logistic_conf_mat.png',dpi=300)

fpr, tpr, thresholds = roc_curve(y_test, y_probs)
auc_score = roc_auc_score(y_test, y_probs)

plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray') 
plt.xlabel('Tasa de falsos positivos (FPR)')
plt.ylabel('Tasa de verdaderos positivos (TPR)')
plt.title('Curva ROC - Regresión Logística')
plt.legend(loc='lower right')
plt.grid()
plt.show()
plt.savefig('ROC_Regression_Logistica.png', dpi=300)

# Random forest
model_rf = tunning_rf_model(X_train, y_train, threshold,optimization) 
print("Sensibilidad:", f"{recall_score(y_test, y_pred)*100:.2f}%")
print("Exactitud:",f"{accuracy_score(y_test, y_pred)*100:.2f}%")
print("F1 scorer:", f"{f1_score(y_test, y_pred)*100:.2f}%")
print("DOR:",class_likelihood_ratios(y_test, y_pred, labels=[0, 1])) # Uncomment this line to perform hyperparameter tuning
