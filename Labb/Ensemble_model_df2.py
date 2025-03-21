from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import ast
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os

"""
Även detta var väldigt nytt för mig så som den andra py-filen. Ingen kodrad är kopierad.
Har även här tagit hjälp från hemsidor för inspiration. ex. från scikit-learn,
stackoverflow samt felsökning från chatgpt.
"""


def get_model_params(model_results_df, model_name):

    # Filtrera dataframe för att hitta rätt modell
    model_row = model_results_df[model_results_df['Model'] == model_name]
    
    if len(model_row) == 0:
        raise ValueError(f"Modell '{model_name}' hittades inte i resultatfilen")
    
    # Hämta parametersträngen och konvertera till ordbok med ast.literal_eval
    params_str = model_row['Best Parameters'].values[0]
    params = ast.literal_eval(params_str)
    
    return params

def train_ensemble_model_df2(X2_train_minmax, X2_val_minmax, X2_test_minmax, 
                            y2_train, y2_val, y2_test, 
                            save_results=True, filename='ensemble_model_results_df2.csv',
                            model_results_file='model_results.csv'):

    # Kombinera tränings- och valideringsdata
    X_train_val = np.vstack((X2_train_minmax, X2_val_minmax))
    y_train_val = np.concatenate((y2_train, y2_val))
    
    # Läs in modellparametrar från CSV-filen
    model_results = pd.read_csv(model_results_file)
    
    # Hämta parametrar för varje modell med hjälpfunktionen
    rf_params = get_model_params(model_results, 'Random Forest Normalization (df2)')
    lr_params = get_model_params(model_results, 'Logistic Regression Normalization (df2)')
    gb_params = get_model_params(model_results, 'Gradient Boosting Normalization (df2)')
    knn_params = get_model_params(model_results, 'KNN Normalization (df2)')
    
    # Skapa individuella modeller med bästa parametrar från CSV-filen
    rf_model = RandomForestClassifier(
        max_depth=rf_params['max_depth'],
        min_samples_leaf=rf_params['min_samples_leaf'],
        min_samples_split=rf_params['min_samples_split'],
        n_estimators=rf_params['n_estimators'],
        random_state=42
    )
    
    lr_model = LogisticRegression(
        C=lr_params['C'],
        penalty=lr_params['penalty'],
        solver=lr_params['solver'],
        random_state=42,
        max_iter=1000
    )
    
    gb_model = GradientBoostingClassifier(
        learning_rate=gb_params['learning_rate'],
        max_depth=gb_params['max_depth'],
        n_estimators=gb_params['n_estimators'],
        random_state=42
    )
    
    knn_model = KNeighborsClassifier(
        n_neighbors=knn_params['n_neighbors'],
        p=knn_params['p'],
        weights=knn_params['weights']
    )
    
    # Träna individuella modeller på kombinerad tränings- och valideringsdata
    rf_model.fit(X_train_val, y_train_val)
    lr_model.fit(X_train_val, y_train_val)
    gb_model.fit(X_train_val, y_train_val)
    knn_model.fit(X_train_val, y_train_val)
    
    # Skapa ensemble-modell med VotingClassifier
    ensemble_model = VotingClassifier(
        estimators=[
            ('rf', rf_model),
            ('lr', lr_model),
            ('gb', gb_model),
            ('knn', knn_model)
        ],
        voting='soft'  # Använd sannolikheter för röstning
    )
    
    # Träna ensemble-modellen på kombinerad tränings- och valideringsdata
    ensemble_model.fit(X_train_val, y_train_val)
    
    # Utvärdera modellerna på testdata
    rf_pred = rf_model.predict(X2_test_minmax)
    lr_pred = lr_model.predict(X2_test_minmax)
    gb_pred = gb_model.predict(X2_test_minmax)
    knn_pred = knn_model.predict(X2_test_minmax)
    ensemble_pred = ensemble_model.predict(X2_test_minmax)
    
    # Beräkna noggrannhet
    rf_accuracy = accuracy_score(y2_test, rf_pred)
    lr_accuracy = accuracy_score(y2_test, lr_pred)
    gb_accuracy = accuracy_score(y2_test, gb_pred)
    knn_accuracy = accuracy_score(y2_test, knn_pred)
    ensemble_accuracy = accuracy_score(y2_test, ensemble_pred)
    
    # Skapa resultatdataframe
    if save_results:
        results = {
            'Modell': ['Random Forest', 'Logistic Regression', 'Gradient Boosting', 'KNN', 'Ensemble (Voting)'],
            'Noggrannhet': [rf_accuracy, lr_accuracy, gb_accuracy, knn_accuracy, ensemble_accuracy]
        }
        
        results_df = pd.DataFrame(results)
        results_df.to_csv(filename, index=False)
        print(f"Resultat sparade i {filename}")
        
        # Skriv ut resultat
        print("\nResultat på testdata:")
        print(f"Random Forest noggrannhet: {rf_accuracy:.4f}")
        print(f"Logistic Regression noggrannhet: {lr_accuracy:.4f}")
        print(f"Gradient Boosting noggrannhet: {gb_accuracy:.4f}")
        print(f"KNN noggrannhet: {knn_accuracy:.4f}")
        print(f"Ensemble (Voting) noggrannhet: {ensemble_accuracy:.4f}")
    
    return ensemble_model

def evaluate_models(X_test, y_test, models, model_names, plot_confusion_matrix=True, save_plots=False, output_dir=None):
    
    if save_plots and output_dir is None:
        output_dir = "."  # Använd aktuell katalog om ingen anges
    
    if save_plots and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    reports = {}
    
    for model, name in zip(models, model_names):
        # Gör prediktioner
        y_pred = model.predict(X_test)
        
        # Skapa classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        reports[name] = report
        
        # Skriv ut classification report
        print(f"\nClassification Report för {name}:")
        print(classification_report(y_test, y_pred))
        
        # Skapa och visa confusion matrix
        if plot_confusion_matrix:
            cm = confusion_matrix(y_test, y_pred)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm)
            
            plt.figure(figsize=(8, 6))
            disp.plot(cmap='Blues')
            plt.title(f'Confusion Matrix - {name}')
            
            if save_plots:
                plt.savefig(os.path.join(output_dir, f"confusion_matrix_{name.replace(' ', '_').lower()}.png"))
            
            plt.show()
    
    return reports
