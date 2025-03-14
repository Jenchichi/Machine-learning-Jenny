from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 


# Skapar en funktion för att utföra GridsearchCV
def perform_grid_search(model, param_grid, X_train, y_train, X_val, y_val, X_test, y_test, model_name, cv=5, scoring='accuracy'):
    """
    Utför GridSearchCV för en given modell och parametergrid, tränar modellen,
    gör förutsägelser på valideringsdata och beräknar utvärderingsmått.
    
    :param model: Maskininlärningsmodellen (t.ex. RandomForestClassifier).
    :param param_grid: Dictionary med hyperparametrar att testa.
    :param X_train: Träningsdata (features).
    :param y_train: Träningsdata (target).
    :param X_val: Valideringsdata (features).
    :param y_val: Valideringsdata (target).
    :param X_test: Testdata (features).
    :param y_test: Testdata (target).
    :param cv: Antal folds för korsvalidering (default: 5).
    :param scoring: Mätvärde för att välja bästa modell (default: 'accuracy').
    :return: Den bästa modellen, dess parametrar och utvärderingsmått.
    """
    # Utför GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)
    
    # Hämta den bästa modellen och dess parametrar
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Gör förutsägelser på valideringsdata
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    
    # Beräkna utvärderingsmått
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)
    
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    # Skapa resultaten och spara i en csv fil
    evaluation_metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }
    
    return best_model, best_params, evaluation_metrics


# Hyperperametrar
# Definiera parametergrid för RandomForest
param_grid_rf = {
    'n_estimators': [10, 20, 30],  # Antal träd i skogen
    'max_depth': [5, 8, 10],  # Maxdjup för varje träd
    'min_samples_split': [10, 15, 20],  # Minsta antal sampel för att dela en nod
    'min_samples_leaf': [2, 3, 4, 5]  # Minsta antal sampel i ett löv
}

# Definiera parametergrid, för LogisticRegression
param_grid_lr = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100], #Regulariseringsstyrka
    'penalty': ['l1', 'l2'], # Strafftyp
    'solver': ['saga'] # Optimeringsteknik
}

# Definiera parametergrid för GradienBoostingClassifier
param_grid_gb = {
    'n_estimators': [10, 20, 30], # Antal träd
    'learning_rate': [0.01, 0.1, 1], # Inlärningshastighet
    'max_depth': [3, 5, 10] # Maxdjup för träd
}

# Definiera parametergrid för KneighborsClassifier
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],  # Antal grannar
    'weights': ['uniform', 'distance'],  # Viktning av grannar: 'uniform' eller 'distance'
    'p': [1,2] # Avståndsmått: 1 för Manhattan, 2 för Euklidiskt avstånd
}


# Maskininlärnings modeller
def train_and_evaluate_models(X1_train_standard, X1_val_standard, X1_test_standard, X1_train_minmax, X1_val_minmax, X1_test_minmax, 
                              X2_train_standard, X2_val_standard, X2_test_standard, X2_train_minmax, X2_val_minmax, X2_test_minmax, 
                              y1_train, y1_val, y1_test, y2_train, y2_val, y2_test):

    # Skapa modell för RandomForestClassifier:
    rf_model = RandomForestClassifier(random_state=42)
    # Träna och utvärdera RandomForest på df1 - Standardization
    best_rf_model_df1, best_rf_params_df1, rf_metrics_df1 = perform_grid_search(
        model=rf_model,
        param_grid=param_grid_rf,
        X_train=X1_train_standard,
        y_train=y1_train,
        X_val=X1_val_standard,
        y_val=y1_val,
        X_test=X1_test_standard,
        y_test=y1_test,
        model_name="Random Forest Standardization (df1)"
    )

    # Träna och utvärdera RandomForest på df2 - Standardization
    best_rf_model_df2, best_rf_params_df2, rf_metrics_df2 = perform_grid_search(
        model=rf_model,
        param_grid=param_grid_rf,
        X_train=X2_train_standard,
        y_train=y2_train,
        X_val=X2_val_standard,
        y_val=y2_val,
        X_test=X2_test_standard,
        y_test=y2_test,
        model_name="Random Forest Standardization (df2)"
    )

    # Träna och utvärdera RandomForest på df1 - Normalization
    best_rf_model_df1_minmax, best_rf_params_df1_minmax, rf_metrics_df1_minmax = perform_grid_search(
        model=rf_model,
        param_grid=param_grid_rf,
        X_train=X1_train_minmax,
        y_train=y1_train,
        X_val=X1_val_minmax,
        y_val=y1_val,
        X_test=X1_test_minmax,
        y_test=y1_test,
        model_name="Random Forest Normalization (df1)"
    )

    # Träna och utvärdera RandomForest på df2 - Normalization
    best_rf_model_df2_minmax, best_rf_params_df2_minmax, rf_metrics_df2_minmax = perform_grid_search(
        model=rf_model,
        param_grid=param_grid_rf,
        X_train=X2_train_minmax,
        y_train=y2_train,
        X_val=X2_val_minmax,
        y_val=y2_val,
        X_test=X2_test_minmax,
        y_test=y2_test,
        model_name="Random Forest Normalization (df2)"
    )


    # Skapa modell
    lr_model = LogisticRegression(random_state=42)

    # Träna och utvärdera Logistic Regression på df1 - Standardization
    best_lr_model_df1, best_lr_params_df1, lr_metrics_df1 = perform_grid_search(
        model=lr_model,
        param_grid=param_grid_lr,
        X_train=X1_train_standard,
        y_train=y1_train,
        X_val=X1_val_standard,
        y_val=y1_val,
        X_test=X1_test_standard,
        y_test=y1_test,
        model_name="Logistic Regression Standardization (df1)"
    )

    # Träna och utvärdera Logistic Regression på df2 - Standardization
    best_lr_model_df2, best_lr_params_df2, lr_metrics_df2 = perform_grid_search(
        model=lr_model,
        param_grid=param_grid_lr,
        X_train=X2_train_standard,
        y_train=y2_train,
        X_val=X2_val_standard,
        y_val=y2_val,
        X_test=X2_test_standard,
        y_test=y2_test,
        model_name="Logistic Regression Standardization (df2)"
    )

    # Träna och utvärdera Logistic Regression på df2 - Normalization
    best_lr_model_df1_minmax, best_lr_params_df1_minmax, lr_metrics_df1_minmax = perform_grid_search(
        model=lr_model,
        param_grid=param_grid_lr,
        X_train=X1_train_minmax,
        y_train=y1_train,
        X_val=X1_val_minmax,
        y_val=y1_val,
        X_test=X1_test_minmax,
        y_test=y1_test,
        model_name="Logistic Regression Normalization (df1)"
    )

    # Träna och utvärdera Logistic Regression på df2 - Normalization
    best_lr_model_df2_minmax, best_lr_params_df2_minmax, lr_metrics_df2_minmax = perform_grid_search(
        model=lr_model,
        param_grid=param_grid_lr,
        X_train=X2_train_minmax,
        y_train=y2_train,
        X_val=X2_val_minmax,
        y_val=y2_val,
        X_test=X2_test_minmax,
        y_test=y2_test,
        model_name="Logistic Regression Normalization (df2)"
    )


    # Skapa modell
    gb_model = GradientBoostingClassifier(random_state=42)

    # Träna och utvärdera GradientBoostingClassifier på df1 - Standardization
    best_gb_model_df1, best_gb_params_df1, gb_metrics_df1 = perform_grid_search(
        model=gb_model,
        param_grid=param_grid_gb,
        X_train=X1_train_standard,
        y_train=y1_train,
        X_val=X1_val_standard,
        y_val=y1_val,
        X_test=X1_test_standard,
        y_test=y1_test,
        model_name="Gradient Boosting Standardization (df1)"
    )

    # Träna och utvärdera GradientBoostingClassifier på df2 - Standardization
    best_gb_model_df2, best_gb_params_df2, gb_metrics_df2 = perform_grid_search(
        model=gb_model,
        param_grid=param_grid_gb,
        X_train=X2_train_standard,
        y_train=y2_train,
        X_val=X2_val_standard,
        y_val=y2_val,
        X_test=X2_test_standard,
        y_test=y2_test,
        model_name="Gradient Boosting Standardization (df2)"
    )

    # Träna och utvärdera GradientBoostingClassifier på df1 - Normalization
    best_gb_model_df1_minmax, best_gb_params_df1_minmax, gb_metrics_df1_minmax = perform_grid_search(
        model=gb_model,
        param_grid=param_grid_gb,
        X_train=X1_train_minmax,
        y_train=y1_train,
        X_val=X1_val_minmax,
        y_val=y1_val,
        X_test=X1_test_minmax,
        y_test=y1_test,
        model_name="Gradient Boosting Normalization (df1)"
    )

    # Träna och utvärdera GradientBoostingClassifier på df2 - Normalization
    best_gb_model_df2_minmax, best_gb_params_df2_minmax, gb_metrics_df2_minmax = perform_grid_search(
        model=gb_model,
        param_grid=param_grid_gb,
        X_train=X2_train_minmax,
        y_train=y2_train,
        X_val=X2_val_minmax,
        y_val=y2_val,
        X_test=X2_test_minmax,
        y_test=y2_test,
        model_name="Gradient Boosting Normalization (df2)"
    )


    # Skapa modell
    knn_model = KNeighborsClassifier()

    # Träna och utvärdera KNeighborsClassifier på df1 - Standardization
    best_knn_model_df1, best_knn_params_df1, knn_metrics_df1 = perform_grid_search(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X1_train_standard,
        y_train=y1_train,
        X_val=X1_val_standard,
        y_val=y1_val,
        X_test=X1_test_standard,
        y_test=y1_test,
        model_name="KNN Standardization (df1)"
    )

    # Träna och utvärdera KNeighborsClassifier på df2 - Standardization
    best_knn_model_df2, best_knn_params_df2, knn_metrics_df2 = perform_grid_search(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X2_train_standard,
        y_train=y2_train,
        X_val=X2_val_standard,
        y_val=y2_val,
        X_test=X2_test_standard,
        y_test=y2_test,
        model_name="KNN Standardization (df2)"
    )

    # Träna och utvärdera KNeighborsClassifier på df1 - Normalization
    best_knn_model_df1_minmax, best_knn_params_df1_minmax, knn_metrics_df1_minmax = perform_grid_search(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X1_train_minmax,
        y_train=y1_train,
        X_val=X1_val_minmax,
        y_val=y1_val,
        X_test=X1_test_minmax,
        y_test=y1_test,
        model_name="KNN Normalization (df1)"
    )

    # Träna och utvärdera KNeighborsClassifier på df2 - Normalization
    best_knn_model_df2_minmax, best_knn_params_df2_minmax, knn_metrics_df2_minmax = perform_grid_search(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X2_train_minmax,
        y_train=y2_train,
        X_val=X2_val_minmax,
        y_val=y2_val,
        X_test=X2_test_minmax,
        y_test=y2_test,
        model_name="KNN Normalization (df2)"
    )



