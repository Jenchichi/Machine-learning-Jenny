from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 


# Skapar en funktion för att utföra GridsearchCV
def perform_grid_search(model, param_grid, X_train, y_train, X_val, y_val, model_name, cv=5, scoring='accuracy'):
    """
    Utför GridSearchCV för en given modell och parametergrid, tränar modellen,
    gör förutsägelser på valideringsdata och beräknar utvärderingsmått.
    
    :param model: Maskininlärningsmodellen (t.ex. RandomForestClassifier).
    :param param_grid: Dictionary med hyperparametrar att testa.
    :param X_train: Träningsdata (features).
    :param y_train: Träningsdata (target).
    :param X_val: Valideringsdata (features).
    :param y_val: Valideringsdata (target).
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
    
    # Beräkna utvärderingsmått
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    
    # Skapa en dictionary med resultaten
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
    'n_estimators': [30, 50, 100],  # Antal träd i skogen
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
def train_and_evaluate_models(X1_train_standard, X1_val_standard, X1_train_minmax, X1_val_minmax, 
                              X2_train_standard, X2_val_standard, X2_train_minmax, X2_val_minmax, 
                              y1_train, y1_val, y2_train, y2_val):
    """
    Tränar och utvärderar alla modeller på både standardiserad och normaliserad data för båda dataset.
    
    :param X1_train_standard: Träningsdata för df1 (standardiserad).
    :param X1_val_standard: Valideringsdata för df1 (standardiserad).
    :param X1_train_minmax: Träningsdata för df1 (normaliserad).
    :param X1_val_minmax: Valideringsdata för df1 (normaliserad).
    :param X2_train_standard: Träningsdata för df2 (standardiserad).
    :param X2_val_standard: Valideringsdata för df2 (standardiserad).
    :param X2_train_minmax: Träningsdata för df2 (normaliserad).
    :param X2_val_minmax: Valideringsdata för df2 (normaliserad).
    :param y1_train: Träningsdata (target) för df1.
    :param y1_val: Valideringsdata (target) för df1.
    :param y2_train: Träningsdata (target) för df2.
    :param y2_val: Valideringsdata (target) för df2.
    :return: En dictionary med resultaten för alla modeller.
    """
    results = {}

    # RandomForest
    rf_model = RandomForestClassifier(random_state=42)
    results['rf_standard_df1'] = perform_grid_search(rf_model, param_grid_rf, X1_train_standard, y1_train, X1_val_standard, y1_val, "Random Forest Standardization (df1)")
    results['rf_standard_df2'] = perform_grid_search(rf_model, param_grid_rf, X2_train_standard, y2_train, X2_val_standard, y2_val, "Random Forest Standardization (df2)")
    results['rf_minmax_df1'] = perform_grid_search(rf_model, param_grid_rf, X1_train_minmax, y1_train, X1_val_minmax, y1_val, "Random Forest Normalization (df1)")
    results['rf_minmax_df2'] = perform_grid_search(rf_model, param_grid_rf, X2_train_minmax, y2_train, X2_val_minmax, y2_val, "Random Forest Normalization (df2)")

    # LogisticRegression
    lr_model = LogisticRegression(random_state=42)
    results['lr_standard_df1'] = perform_grid_search(lr_model, param_grid_lr, X1_train_standard, y1_train, X1_val_standard, y1_val, "Logistic Regression Standardization (df1)")
    results['lr_standard_df2'] = perform_grid_search(lr_model, param_grid_lr, X2_train_standard, y2_train, X2_val_standard, y2_val, "Logistic Regression Standardization (df2)")
    results['lr_minmax_df1'] = perform_grid_search(lr_model, param_grid_lr, X1_train_minmax, y1_train, X1_val_minmax, y1_val, "Logistic Regression Normalization (df1)")
    results['lr_minmax_df2'] = perform_grid_search(lr_model, param_grid_lr, X2_train_minmax, y2_train, X2_val_minmax, y2_val, "Logistic Regression Normalization (df2)")

    # GradientBoosting
    gb_model = GradientBoostingClassifier(random_state=42)
    results['gb_standard_df1'] = perform_grid_search(gb_model, param_grid_gb, X1_train_standard, y1_train, X1_val_standard, y1_val, "Gradient Boosting Standardization (df1)")
    results['gb_standard_df2'] = perform_grid_search(gb_model, param_grid_gb, X2_train_standard, y2_train, X2_val_standard, y2_val, "Gradient Boosting Standardization (df2)")
    results['gb_minmax_df1'] = perform_grid_search(gb_model, param_grid_gb, X1_train_minmax, y1_train, X1_val_minmax, y1_val, "Gradient Boosting Normalization (df1)")
    results['gb_minmax_df2'] = perform_grid_search(gb_model, param_grid_gb, X2_train_minmax, y2_train, X2_val_minmax, y2_val, "Gradient Boosting Normalization (df2)")

    # KNeighbors
    knn_model = KNeighborsClassifier()
    results['knn_standard_df1'] = perform_grid_search(knn_model, param_grid_knn, X1_train_standard, y1_train, X1_val_standard, y1_val, "KNN Standardization (df1)")
    results['knn_standard_df2'] = perform_grid_search(knn_model, param_grid_knn, X2_train_standard, y2_train, X2_val_standard, y2_val, "KNN Standardization (df2)")
    results['knn_minmax_df1'] = perform_grid_search(knn_model, param_grid_knn, X1_train_minmax, y1_train, X1_val_minmax, y1_val, "KNN Normalization (df1)")
    results['knn_minmax_df2'] = perform_grid_search(knn_model, param_grid_knn, X2_train_minmax, y2_train, X2_val_minmax, y2_val, "KNN Normalization (df2)")

    return results

