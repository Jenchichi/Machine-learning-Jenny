from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import os


# Skapar en funktion för att utföra GridsearchCV
def perform_grid_search(model, param_grid, X_train, y_train, X_val, y_val, X_test, y_test, model_name, cv=5, scoring='accuracy', save_results=False, filename='model_results.csv'):

    # Utför GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, scoring=scoring)
    grid_search.fit(X_train, y_train)
    
    print(f"Bästa parametrar för: {model_name}:", grid_search.best_params_)

    # Hämta den bästa modellen och dess parametrar
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Gör förutsägelser på validerings- och testdata
    y_val_pred = best_model.predict(X_val)
    y_test_pred = best_model.predict(X_test)
    
    # Beräkna utvärderingsmått (endast accuracy)
    val_accuracy = accuracy_score(y_val, y_val_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    # Spara resultaten om save_results är True
    if save_results:
        results = {
            'Model': [model_name],
            'Best Parameters': [str(best_params)],
            'Validation Accuracy': [val_accuracy],
            'Test Accuracy': [test_accuracy]
        }
        results_df = pd.DataFrame(results)

        # Kontrollera om filen redan finns
        if os.path.exists(filename):
            existing_df = pd.read_csv(filename)
            # Kontrollera om modellen redan finns i filen
            if model_name not in existing_df['Model'].values:
                # Lägg till nya resultat om modellen inte redan finns
                results_df.to_csv(filename, mode='a', header=False, index=False)
                print(f"Nya resultat för '{model_name}' sparades i {filename}.")
            else:
                print(f"Modellen '{model_name}' finns redan i filen. Inga nya resultat sparades.")
        else:
            # Skapa en ny fil om den inte finns
            results_df.to_csv(filename, mode='w', header=True, index=False)
            print(f"Ny fil skapad: {filename}")
    
    return best_model, best_params, val_accuracy, test_accuracy


# Hyperperametrar
# Definiera parametergrid för RandomForest
param_grid_rf = {
    'n_estimators': [10, 20, 30],  # Antal träd i skogen
    'max_depth': [5, 8, 10],  # Maxdjup för varje träd
    'min_samples_split': [10, 15, 20],  # Minsta antal sampel för att dela en nod
    'min_samples_leaf': [2, 3, 4, 5] # Minsta antal sampel i ett löv
}

# Definiera parametergrid, för LogisticRegression
param_grid_lr = {
    'C': [0.1, 0.5, 1, 5, 10, 100], #Regulariseringsstyrka
    'penalty': ['l1', 'l2'], # Strafftyp
    'solver': ['saga'] # Optimeringsteknik
}

# Definiera parametergrid för GradientBoostingClassifier
param_grid_gb = {
    'n_estimators': [10, 20, 30], # Antal träd
    'learning_rate': [0.001, 0.01, 0.1, 1], # Inlärningshastighet
    'max_depth': [3, 5, 7, 10] # Maxdjup för träd
}

# Definiera parametergrid för KneighborsClassifier
param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9],  # Antal grannar
    'weights': ['uniform', 'distance'],  # Viktning av grannar: 'uniform' eller 'distance'
    'p': [1, 2] # Avståndsmått: 1 för Manhattan, 2 för Euklidiskt avstånd
}


# Maskininlärnings modeller
def train_and_evaluate_models(X1_train_minmax, X1_val_minmax, X1_test_minmax, 
                              X2_train_minmax, X2_val_minmax, X2_test_minmax, 
                              y1_train, y1_val, y1_test, y2_train, y2_val, y2_test):

    # Skapa modell för RandomForestClassifier:
    rf_model = RandomForestClassifier(random_state=42)

    # Träna och utvärdera RandomForest på df1 - Normalization
    best_rf_model_df1_minmax, best_rf_params_df1_minmax, val_accuracy_df1_minmax, test_accuracy_df1_minmax = perform_grid_search(
        model=rf_model,
        param_grid=param_grid_rf,
        X_train=X1_train_minmax,
        y_train=y1_train,
        X_val=X1_val_minmax,
        y_val=y1_val,
        X_test=X1_test_minmax,
        y_test=y1_test,
        model_name="Random Forest Normalization (df1)",
        save_results = True,
        filename = "model_results.csv"
    )

    # Träna och utvärdera RandomForest på df2 - Normalization
    best_rf_model_df2_minmax, best_rf_params_df2_minmax, val_accuracy_df2_minmax, test_accuracy_df2_minmax = perform_grid_search(
        model=rf_model,
        param_grid=param_grid_rf,
        X_train=X2_train_minmax,
        y_train=y2_train,
        X_val=X2_val_minmax,
        y_val=y2_val,
        X_test=X2_test_minmax,
        y_test=y2_test,
        model_name="Random Forest Normalization (df2)",
        save_results = True,
        filename = "model_results.csv"
    )


    # Skapa modell
    lr_model = LogisticRegression(random_state=42)

    # Träna och utvärdera Logistic Regression på df1 - Normalization
    best_lr_model_df1_minmax, best_lr_params_df1_minmax, val_accuracy_df1_minmax, test_accuracy_df1_minmax = perform_grid_search(
        model=lr_model,
        param_grid=param_grid_lr,
        X_train=X1_train_minmax,
        y_train=y1_train,
        X_val=X1_val_minmax,
        y_val=y1_val,
        X_test=X1_test_minmax,
        y_test=y1_test,
        model_name="Logistic Regression Normalization (df1)",
        save_results = True,
        filename = "model_results.csv"
    )

    # Träna och utvärdera Logistic Regression på df2 - Normalization
    best_lr_model_df2_minmax, best_lr_params_df2_minmax, val_accuracy_df2_minmax, test_accuracy_df2_minmax = perform_grid_search(
        model=lr_model,
        param_grid=param_grid_lr,
        X_train=X2_train_minmax,
        y_train=y2_train,
        X_val=X2_val_minmax,
        y_val=y2_val,
        X_test=X2_test_minmax,
        y_test=y2_test,
        model_name="Logistic Regression Normalization (df2)",
        save_results = True,
        filename = "model_results.csv"
    )


    # Skapa modell
    gb_model = GradientBoostingClassifier(random_state=42)

    # Träna och utvärdera GradientBoostingClassifier på df1 - Normalization
    best_gb_model_df1_minmax, best_gb_params_df1_minmax, val_accuracy_df1_minmax, test_accuracy_df1_minmax = perform_grid_search(
        model=gb_model,
        param_grid=param_grid_gb,
        X_train=X1_train_minmax,
        y_train=y1_train,
        X_val=X1_val_minmax,
        y_val=y1_val,
        X_test=X1_test_minmax,
        y_test=y1_test,
        model_name="Gradient Boosting Normalization (df1)",
        save_results = True,
        filename = "model_results.csv"
    )

    # Träna och utvärdera GradientBoostingClassifier på df2 - Normalization
    best_gb_model_df2_minmax, best_gb_params_df2_minmax, val_accuracy_df2_minmax, test_accuracy_df2_minmax = perform_grid_search(
        model=gb_model,
        param_grid=param_grid_gb,
        X_train=X2_train_minmax,
        y_train=y2_train,
        X_val=X2_val_minmax,
        y_val=y2_val,
        X_test=X2_test_minmax,
        y_test=y2_test,
        model_name="Gradient Boosting Normalization (df2)",
        save_results = True,
        filename = "model_results.csv"
    )


    # Skapa modell
    knn_model = KNeighborsClassifier()

    # Träna och utvärdera KNeighborsClassifier på df1 - Normalization
    best_knn_model_df1_minmax, best_knn_params_df1_minmax, val_accuracy_df1_minmax, test_accuracy_df1_minmax = perform_grid_search(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X1_train_minmax,
        y_train=y1_train,
        X_val=X1_val_minmax,
        y_val=y1_val,
        X_test=X1_test_minmax,
        y_test=y1_test,
        model_name="KNN Normalization (df1)",
        save_results = True,
        filename = "model_results.csv"
    )

    # Träna och utvärdera KNeighborsClassifier på df2 - Normalization
    best_knn_model_df2_minmax, best_knn_params_df2_minmax, val_accuracy_df2_minmax, test_accuracy_df2_minmax = perform_grid_search(
        model=knn_model,
        param_grid=param_grid_knn,
        X_train=X2_train_minmax,
        y_train=y2_train,
        X_val=X2_val_minmax,
        y_val=y2_val,
        X_test=X2_test_minmax,
        y_test=y2_test,
        model_name="KNN Normalization (df2)",
        save_results = True,
        filename = "model_results.csv"
    )

