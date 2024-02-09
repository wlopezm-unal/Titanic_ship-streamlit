
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,f1_score
from sklearn.metrics import accuracy_score
import upload_clean_data
import pandas as pd

####--------------------Create predict model---------------------------------------#
def data():
    #----------------------------------upload data already cleaning------------------#
    #---Upload_clean_data it do the process the upload and cleaning data
    df_train=upload_clean_data.create_db('G:/carpeta/Documentos/Programación/Spaceship_titanic_/train.csv')
    X=df_train.iloc[:,:-1]
    y=df_train.iloc[:,-1]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def predictGnb():
    
    X_train, X_test, y_train, y_test=data()

    # Inicializar el clasificador Naive Bayes
    model = GaussianNB()

    # Define the hyperparameters to tune

    #It various parameters were used to calculate the best score from 'var_smoothing': [1e-9, 1e-8, 1e-7] and  'priors': [None, [0.25, 0.75], [0.5, 0.5]]
    #where  found that  priors': [0.5, 0.5], 'var_smoothing': 1e-08} are the best parametres to this model
    param_grid = {
        'var_smoothing': [ 1e-7],
        'priors': [[0.5, 0.5]],  
        
    }
    
    # Perform grid search to find the best hyperparameters
    grid_search = GridSearchCV(model, param_grid, cv=10)
    #entrenar el modelo
    grid_search.fit(X_train, y_train)

    # Realizar predicciones
    y_pred_naive = grid_search.predict(X_test)

    # Realizar validación cruzada para obtener una estimación más precisa del rendimiento del modelo
    cv_scores = cross_val_score(model, X_train, y_train, cv=10)
    cv_accuracy = cv_scores.mean()
    cv_f1score = cv_scores.mean()

    # Compute the accuracy
    accuracy= accuracy_score(y_test, y_pred_naive)
    f1score=f1_score(y_test, y_pred_naive)


    return {"Accuracy":accuracy, "F1_score": f1score, "cross validation accuracy":cv_accuracy , "cross validation f1 score": cv_f1score, }
    #return best_params


def LRM():
    X_train, X_test, y_train, y_test=data()
    # Crear el modelo de regresión logística
    model_logistic=LogisticRegression()

    # Definir los hiperparámetros a ajustar
    #It various parameters were used to calculate the best score from 'C': [0.001, 0.01, 0.1, 1, 10, 100], solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag', 'saga']
    #'fit_intercept': [True, False] y "max_iter": [10000], where  found that {'C': 0.01, 'fit_intercept': True, 'max_iter': 10000, 'solver': 'lbfgs'} are the best parametre to our model
    param_grid = {

        'C': [ 0.01],
        'solver': ['lbfgs'],
        'fit_intercept': [True],
        "max_iter": [10000]
    }
    # Realizar la búsqueda en cuadrícula para evaluar los hiperparámetros
    grid_search = GridSearchCV(model_logistic, param_grid, cv=10)
    grid_search.fit(X_train, y_train)

    # Obtener el mejor modelo con los mejores hiperparámetros
    #best_params = grid_search.best_params_

    # Realizar validación cruzada para obtener una estimación más precisa del rendimiento del modelo
    cv_scores = cross_val_score(model_logistic, X_train, y_train, cv=10)
    cv_accuracy = cv_scores.mean()
    cv_f1score = cv_scores.mean()

    # Realizar predicciones en el conjunto de prueba
    y_pred = grid_search.predict(X_test)
    accuracy_LRM=accuracy_score(y_test, y_pred)
    f1score_LRM=f1_score(y_test, y_pred, average="weighted")

    return {"Model": "LogisticRegression" , "Accuracy":accuracy_LRM, "F1_score": f1score_LRM, "cross validation accuracy":cv_accuracy , "cross validation f1 score": cv_f1score}
    #return f'los mejores parametros son : {best_params}'

def random_forest(data_input):
    X_train, X_test, y_train, y_test=data()
    # Crear el modelo de regresión logística

    # Inicializar el clasificador Random Forest
    clf = RandomForestClassifier()
    # Definir los hiperparámetros a evaluar
    #It various parameters were used to calculate the best score from 'n_estimators': [100, 200, 300],max_depth': [10, 20, 30], min_samples_split': [2, 4, 8, 10], 
    #'min_samples_leaf': [1, 2, 4, 5 ]
    #'where  found that {'max_depth': 30, 'min_samples_leaf': 5, 'min_samples_split': 4, 'n_estimators': 200} are best parametre to this model
    param_grid = {
        'n_estimators': [200],
        'max_depth': [30],
        'min_samples_split': [4],
        'min_samples_leaf': [ 5 ]
        
    }
    # Realizar la búsqueda en cuadrícula para encontrar los mejores hiperparámetros
    grid_search = GridSearchCV(clf, param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    # Use the best model to make predictions
    #best_model = grid_search.best_estimator_
    #best_params = grid_search.best_params_
    predictions = grid_search.predict(data_input)
    # Compute the accuracy
    #accuracy= accuracy_score(y_test, predictions)
    #f1score=f1_score(y_test, predictions)
    if predictions==True:
        passenger="The passager was salved"
    else:
        passenger="The passager was dead"
    return {"Model": "Random Forest" , "Predict":predictions, "Passenger": passenger}
    #return  f'los mejores parametros son : {best_params}'

# item={"PassengerId": "0001_01", "HomePlanet": "Europa", "CryoSleep": False, "Cabin": "B/0/P", "Destination": "TRAPPIST-1e", "Age": 39.0, "VIP": False, "RoomService": 0.0, "FoodCourt": 0.0, "ShoppingMall": 0.0, "Spa": 0.0, "VRDeck": 0.0, "Name": "Maham Ofracculy"}
# df=pd.DataFrame(item, index=pd.Index([0])) 
# data_input=upload_clean_data.input_data(df)
# prediccion_1=random_forest(data_input.iloc[:,:])
# print(prediccion_1)