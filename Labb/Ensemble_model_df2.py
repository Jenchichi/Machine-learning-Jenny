from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd 
import os

# Ensemble modell med VotingClassifier
def create_ensemble_model(X_train, y_train, X_val, y_val, X_test, y_test, dataset_name="df2"):
    """
    Skapar en ensemble-modell med VotingClassifier baserat på de bästa parametrarna
    från tidigare träning. Använder 'hard' voting (majoritetsröstning).
    
    Args:
        X_train, y_train: Träningsdata
        X_val, y_val: Valideringsdata
        X_test, y_test: Testdata
        dataset_name: Namnet på datasetet (default: "df2")
    
    Returns:
        ensemble_model: Den tränade ensemble-modellen
        accuracy: Noggrannhet på testdata
    """
    # Ladda resultat från CSV-filen
    results_df = pd.read_csv('model_results.csv')
    
    # Filtrera resultat specifikt för df2
    df2_results = results_df[results_df['Model'].str.contains(dataset_name)]
    
    # Extrahera bästa parametrar för varje modell
    rf_params = eval(df2_results[df2_results['Model'].str.contains('Random Forest')]['Best Parameters'].values[0])
    lr_params = eval(df2_results[df2_results['Model'].str.contains('Logistic Regression')]['Best Parameters'].values[0])
    gb_params = eval(df2_results[df2_results['Model'].str.contains('Gradient Boosting')]['Best Parameters'].values[0])
    knn_params = eval(df2_results[df2_results['Model'].str.contains('KNN')]['Best Parameters'].values[0])
    
    # Lägg till random_state för modeller som stödjer det
    rf_params['random_state'] = 42
    lr_params['random_state'] = 42
    gb_params['random_state'] = 42
    
    # Skapa modeller med de extraherade parametrarna
    rf_best = RandomForestClassifier(**rf_params)
    lr_best = LogisticRegression(**lr_params)
    gb_best = GradientBoostingClassifier(**gb_params)
    knn_best = KNeighborsClassifier(**knn_params)
    
    # Kombinera tränings- och valideringsdata för att träna den slutliga modellen
    X_train_combined = pd.concat([pd.DataFrame(X_train), pd.DataFrame(X_val)])
    y_train_combined = pd.concat([pd.Series(y_train), pd.Series(y_val)])
    
    # Skapa VotingClassifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_best),
            ('lr', lr_best),
            ('gb', gb_best),
            ('knn', knn_best)
        ],
        voting='hard'  # Använd majoritetsröstning
    )
    
    # Träna ensemble-modellen på kombinerad tränings- och valideringsdata
    print(f"Tränar ensemble-modell med VotingClassifier på {dataset_name}...")
    ensemble_model.fit(X_train_combined, y_train_combined)
    
    # Utvärdera på testdata
    y_test_pred = ensemble_model.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_test_pred)
    
    print(f"Ensemble-modell (VotingClassifier) på {dataset_name}:")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # Spara resultatet i CSV-filen
    results = {
        'Model': [f"Ensemble VotingClassifier ({dataset_name})"],
        'Best Parameters': ["N/A - Ensemble av bästa modeller"],
        'Validation Accuracy': ["N/A - Tränad på kombinerad data"],
        'Test Accuracy': [test_accuracy]
    }
    results_df = pd.DataFrame(results)
    
    filename = "ensemble_model_results_df2.csv"
    if os.path.exists(filename):
        existing_df = pd.read_csv(filename)
        model_name = f"Ensemble VotingClassifier ({dataset_name})"
        if model_name not in existing_df['Model'].values:
            results_df.to_csv(filename, mode='a', header=False, index=False)
            print(f"Nya resultat för ensemble-modell sparades i {filename}.")
        else:
            print(f"Ensemble-modellen finns redan i filen. Inga nya resultat sparades.")
    else:
        results_df.to_csv(filename, mode='w', header=True, index=False)
        print(f"Ny fil skapad: {filename}")
    
    return ensemble_model, test_accuracy