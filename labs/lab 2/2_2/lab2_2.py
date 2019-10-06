# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% [markdown]
# # Laboratorio 2.2: Clasificación
# 
# Bárbara Poblete, Felipe Bravo, Aymé Arango, Juglar Díaz, Hernán Sarmiento, Juan Pablo Silva
# **Septiembre 2019**
#%% [markdown]
# ## =================== INTEGRANTES =====================
# 
# Escriba a continuación el nombre de los integrantes del presente laboratorio:
# 
# 1. 
# 
# 2. 
# 
# ## =====================================================
#%% [markdown]
# # Instrucciones
# 
# 
# 1. El formato de entrega es un documento en **.html**, generado por jupyter.
# 
# 2. El laboratorio debe realizarse en grupos de **2 personas**.
# 
# 3. Asegúrese que están los nombres de los integrantes. Sólo uno de los integrantes debe subir este archivo a U-Cursos antes de finalizar la sesión. 
# 
# 4. Las respuestas a cada pregunta se deben escribir en los bloques que dicen **RESPUESTA A PREGUNTA X.X**.
#%% [markdown]
# # Del Laboratorio 
# 
# En este laboratorio vamos a comparar clasificadores con cierto *baselines* o clasificadores base, y además vamos a trabajar con clases desbalanceadas. 
#%% [markdown]
# # Parte 1: Comparar clasificadores
# 
# Una de las principales tareas en enfoques supervisados es evaluar diferentes clasificadores y encontrar el mejor de alguno de ellos para un problema. Por ejemplo, si tenemos dos (o más) clasificadores y queremos compararlos entre sí, nos interesa responder: *¿Cuál de los clasificadores es el mejor?* 
# Para responder esta pregunta, no existe una única solución. 
# 
# Lo que haremos a continuación será ejecutar diferentes clasificadores y compararlos en base a las métricas de Precision, Recall y F1-score.
#%% [markdown]
# ## Pregunta 1.1  
# 
# Para realizar la evaluación de distintos clasificadores, vamos a crear la función `run_classifier()`, la cual evalúa un clasificador `clf` recibido como parámetro un dataset `X,y` (dividido en training y testing) y un número de tests llamado `num_test`. Esta función almacena y retorna los valores de precision, recall y f1-score en la variable `metrics` además de los resultados de predicción.
# 
# 
# En base a lo anterior, incluya las sentencias que ajusten el modelo junto a su correspondiente predicción sobre los datos. No use cross-validation ni tampoco el parámetro `random_state`.
# 
# 
# ### Respuesta 1.1

#%%
### COMPLETAR ESTE CÓDIGO

## run_classifier recibe un clasificador y un dataset dividido para entrenamiento y testing
## y opcionalmente la cantidad de resultados que se quiere obtener del clasificador

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, recall_score, precision_score


def run_classifier(clf, X, y, num_tests=100):
    metrics = {'f1-score': [], 'precision': [], 'recall': []}
    

    
    for _ in range(num_tests):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30)
        ### INICIO COMPLETAR ACÁ 
        
        #### TIP: en base a los set de entrenamiento, genere la variable predictions 
        #### que contiene las predicciones del modelo
        
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        y_train_pred = clf.predict(X_train)
        
        ### FIN COMPLETAR ACÁ
        
        metrics['y_pred'] = predictions
        metrics['y_prob'] = clf.predict_proba(X_test)[:,1]
        metrics['f1-score'].append(f1_score(y_test, predictions)) 
        metrics['recall'].append(recall_score(y_test, predictions))
        metrics['precision'].append(precision_score(y_test, predictions))
    
    return metrics

#%% [markdown]
# Luego de completar el código anterior, ejecute el siguiente bloque para comparar los distintos clasificadores. 
# Usaremos un **dataset de cáncer de mamas** para evaluar. Información del dataset la puede encontrar en el siguiente link: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html

#%%
## ejecutar este código

from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.svm import SVC  # support vector machine classifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB  # naive bayes
from sklearn.neighbors import KNeighborsClassifier

bc = load_breast_cancer()    # dataset cancer de mamas
X = bc.data
y = bc.target

c0 = ("Base Dummy", DummyClassifier(strategy='stratified'))
c1 = ("Decision Tree", DecisionTreeClassifier())
c2 = ("Gaussian Naive Bayes", GaussianNB())
c3 = ("KNN", KNeighborsClassifier(n_neighbors=5))

classifiers = [c0,c1, c2, c3]

results = {}
for name, clf in classifiers:
    metrics = run_classifier(clf, X, y)   # hay que implementarla en el bloque anterior.
    results[name] = metrics
    print("----------------")
    print("Resultados para clasificador: ",name) 
    print("Precision promedio:",np.array(metrics['precision']).mean())
    print("Recall promedio:",np.array(metrics['recall']).mean())
    print("F1-score promedio:",np.array(metrics['f1-score']).mean())
    print("----------------\n\n")
    

#%% [markdown]
# ### Pregunta 1.2
# 
# Analizando los resultados obtenidos de cada clasificador, y basándose en las métricas calculadas. ¿Cuál es el mejor clasificador? ¿Qué métricas observó para tomar esa decisión y por qué? Fundamente su respuesta.
#%% [markdown]
# ### Respuesta 1.2
# :: En base a los números de rendimiento que se ejecutan dentro del bloque de codigo anterior, el mejor clasificador es Gaussian Naive Bayes, aunque tiene números de rendimiento similares a los de KNN.
# Cabe destacar que el rendimiento del árbol de desición es menos de un 2% peor que el del modelo con mejor ajuste (GNB), por lo que considerando que la interpretabilidad del modelo es mayor a GNB y KNN sería preferible utlizar este clasificador para un problema de cancer de mamas, ya que obtener información para identificar causalidad es relevante en este contexto.
# El indicador considerado como predominante es el F1-score, ya que balancea los aciertos correctos y los negativos correctos, a diferencia de presición y recall que se enfocan en un lado de la predicción, esto es considerando que dentro del contexto del clasificador se espera que el costo de equivocación (predicción errada), sea grande para ambos casos, donde se asigne tratamiento a alguien sano y donde no se asigna a alguien que si lo requiere.
#%% [markdown]
# #Parte 2: Seleccionando hiperparámetros
# Los hiperparámetros son parámetros que no se aprenden directamente dentro de los estimadores. En scikit-learn se pasan como argumentos al constructor de las clases. Por ejemplo que kernel usar para Support Vector Classifier, o que criterion para Decision Tree, etc. Es posible y recomendable buscar en el espacio de hiperparámetros la mejor alternativa. Cualquier parámetro proporcionado al construir un estimador puede optimizarse de esta manera. Para encontrar los nombres y los valores actuales de todos los parámetros para un estimador dado puede usar *estimator.get_params()*.
# 
# Una búsqueda consiste en:
# 
# *   un estimador (regresor o clasificador como sklearn.svm.SVC ());
# *   un espacio de parámetros;
# *   un método para buscar o muestrear candidatos;
# *   un esquema de validación cruzada; y
# *   una función de puntuación(score).
# 
# 
# Tenga en cuenta que es común que un pequeño subconjunto de esos parámetros pueda tener un gran impacto en el rendimiento predictivo o de cálculo del modelo, mientras que otros pueden dejar sus valores predeterminados. Se recomienda leer la documentación de la clase de estimador para obtener una mejor comprensión de su comportamiento esperado, posiblemente leyendo la referencia adjunta a la literatura.
#%% [markdown]
# ###Pregunta 2.1 
# 
# Una alternativa para seleccionar hiperparámetros es GridSearchCV. GridSearchCV considera exhaustivamente todas las combinaciones de parámetros. GridSearchCV recibe un *estimador*, recibe *param_grid* (un diccionario o una lista de diccionarios con los nombres de los parametros a probar como keys y una lista de los valores a probar), *scoring* una o varias funciones de puntuación (score) para evaluar cada combinación de parametros y *cv* una extrategia para hacer validación cruzada.
#%% [markdown]
# El siguiente código muestra como seleccionar el número de vecinos y que pesos otorgar a los vecinos en un clasificador KNN. 
#  
# 

#%%
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.30)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

tuned_parameters = {'n_neighbors': [1,3,5], 'weights': ['uniform','distance']}
score = 'precision'

clf = GridSearchCV(KNeighborsClassifier(), param_grid=tuned_parameters, cv=5,
                       scoring=score)
clf.fit(X_train, y_train)

print("Mejor combinación de parámetros:")
print(clf.best_params_)
 
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))

#%% [markdown]
# ###Pregunta
# *  a) Realice este mismo proceso para un clasificador DecisionTree y los parametros criterion=['gini','entropy'] y max_depth=[1,3,5].
# *  b) ¿Qué puede decir de los resultados, considera que es necesario seguir explorando los parámetros, fue útil hacer este análisis?

#%%
## RESPUESTA A PREGUNTA 2.1 a)

#Completar codigo aca
tuned_parameters = {} #Completar tuned_parameters

#Repetir el codigo de la seccion anterior con KNN pero ahora con decision tree
score = 'precision'

#Construir aca el clf con GridSearch y luego entrenar


print("Mejor combinación de parámetros:")
print(clf.best_params_)
 
y_true, y_pred = y_test, clf.predict(X_test)
print(classification_report(y_true, y_pred))

#%% [markdown]
# ### Respuesta 2.1 b)
# :: 
#%% [markdown]
# ---
# 
# # Parte 3: Tratando con clases desbalanceadas
# 
# Para mejorar el rendimiento de un clasificador sobre clases desbalanceadas existen varias técnicas. En esta parte, veremos cómo tratar con este problema usando (sub/over)sampling de las clases.
# 
# Descargue el dataset `unbalanced.csv` que está en el tutorial. 
# 
# (*Nota: Para ejecutar el siguiente bloque es necesaria la librería `pandas` que viene incluida en Anaconda.*)

#%%
import pandas as pd

# Cargamos dataset desbalanceado
unbalanced = 'unbalanced.csv'
unbalanced = https://users.dcc.uchile.cl/~hsarmien/mineria/datasets/unbalanced.csv

data = pd.read_csv(unbalanced)  # abrimos el archivo csv y lo cargamos en data.
data.head()

#%% [markdown]
# Note el desbalance de las clases ejecutando el siguiente código:

#%%
print("Distribucion de clases original")
data['Class'].value_counts()

#%% [markdown]
# Antes de hacer algo para tratar el desbalance entre las clases debemos antes dividir en train-test.

#%%
data_train, data_test, ytrain, ytest = train_test_split(data, data['Class'], test_size=0.2, stratify=data['Class'])

#%% [markdown]
# Así queda la proporción de clases en el train después de dividir en train-test.

#%%
ytrain.value_counts()

#%% [markdown]
# Ahora, usando el dataset anterior, aplicaremos **oversampling** y **subsampling** al train para que queden balanceados. Ejecute el siguiente código y note ahora que las clases están balanceadas. 

#%%
import numpy as np

print("Distribución de clases usando (over/sub)sampling")
print()

data_train = data_train.reset_index(drop=True)

# oversampling sobre la clase 1
idx = np.random.choice(data_train[data_train['Class'] == 1].index, size=78)
data_oversampled = pd.concat([data_train, data_train.iloc[idx]])
print("Data oversampled on class '1'")
print(data_oversampled['Class'].value_counts())
print()


# subsampling sobre la clase 0
idx = np.random.choice(data_train.loc[data_train.Class == 0].index, size=78, replace=False)
data_subsampled = data_train.drop(data_train.iloc[idx].index)
print("Data subsampled on class '0'")
print(data_subsampled['Class'].value_counts())

#%% [markdown]
# Para la siguiente pregunta, vamos a entrenar un árbol de decisión (`DecisionTreeClassifier`) sobre los 3 datasets por separado (**original**, con **oversampling** y con **subsampling**) y luego comparamos los resultados usando alguna métrica de evaluación.
# 
# Ejecute el siguiente bloque para cargar los datos:

#%%
## ejecutar este código para preparar los datos
from sklearn.metrics import classification_report

# Preparando los data frames para ser compatibles con sklearn

# datos test
X_test = data_test[data_train.columns[:-1]] # todo hasta la penultima columna
y_test = data_test[data_train.columns[-1]]  # la última columna


# datos entrenamiento "originales"
X_orig = data_train[data_train.columns[:-1]] 
y_orig = data_train[data_train.columns[-1]] 

# datos entrenamiento "oversampleados" 
X_over = data_oversampled[data_train.columns[:-1]]
y_over = data_oversampled[data_train.columns[-1]]

# datos entrenamiento "subsampleados"
X_subs = data_subsampled[data_train.columns[:-1]]
y_subs = data_subsampled[data_train.columns[-1]]

#%% [markdown]
# ## Pregunta 3.1
# 
# Complete el código necesario para ejecutar el clasificador en cada uno de los tres casos. Emplee como datos de entrada lo del bloque anterior. Para cada caso entrene con el dataset correspondiente y evalue con el conjunto de test (será el mismo para los tres casos) obtenido con train_test_split sobre los datos originales. 
# 
# Muestre Precision, Recall y F1-score.
# 
#%% [markdown]
# ### RESPUESTA PREGUNTA 3.1 (agregue código en el siguiente bloque)

#%%
## RESPUESTA A PREGUNTA 3.1

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

## Recuerde:
##  - instanciar el clasificador con DecisionTreeClassifier()
##  - entrenar con fit()
##  - hacer las predicciones
##  - Mostrar precision, recall y f1-score.
clf_over = DecisionTreeClassifier()
clf_sub = DecisionTreeClassifier()

clf_over.fit(X_over,y_over)
clf_sub.fit(X_subs,y_subs)

pred_over = clf_over.predict(X_test)
pred_sub = clf_sub.predict(X_test)


#%%
# Aca esta el codigo usando el dataset: original 
print("ORIGINAL::::::::::")
clf_orig = DecisionTreeClassifier()

clf_orig.fit(X_orig,y_orig)
pred_orig = clf_orig.predict(X_test)
print(classification_report(y_test, pred_orig))

# Complete el resto para oversampling y subsampling 


print("OVERSAMPLING::::::::::")
print( classification_report(y_test,pred_over))

print("SUBSAMPLING::::::::::")
print(classification_report(y_test,pred_sub))



#%% [markdown]
# ## Pregunta 3.2
# 
# ¿Cuál estrategia de sampling entrega mejores resultados para la clase minoritaria? 
# 
#%% [markdown]
# ### RESPUESTA A PREGUNTA 3.2
# :: Over sappeling y sub sampeling entregan un mejor resultado que el problema original capturdo por el F1-score, sin embargo, para elegir entre los dos modelos hay un tradeoff entre la presicion y recall, con over y subsampling teniendo mejores resultados respectivamente.
# ::
# 
#%% [markdown]
# ## Pregunta 3.3
# 
# Indique una desventaja de usar oversampling y una desventaja de usar subsampling en clasificación.
# 
#%% [markdown]
# ### RESPUESTA A PREGUNTA 3.3
# :: La principal desventaja de usar over sampleing es tener un mayor costo computacional para entrenar los modelos donde se tiene una base de datos potencialmente muchisimo mas grande, en casos extremos esto podria inducir problemas de memoria o dependiendo del metodo de clasificacion podria generar un tiempo de entrenamiento ordenes de magnitud peor que el de la base original.
# :: La princial desventaja de sub sampeling es perder informacion en el proceso, si es que no se controla que la muestra final tenga la misma distribución que la base original, se puede perder o sesgar los datos con los que se trabaja, lo que evidentemente genera una pérdida de calidad del análisis final. En el caso extremo de subsampling podría perderse una clase completa o suficiente de una como para claificarla dentro de otro grupo, perdiendo por completo un grupo de estudio en el análisis.
# 
# 

