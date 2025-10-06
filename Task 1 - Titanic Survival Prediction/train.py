import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report


# FIXED PATHS
DATA_PATH = r"C:\Users\madhukar\OneDrive\Desktop\CODSOFT\Task 1 - Titanic Survival Prediction\data\Titanic-Dataset.csv"
OUT_DIR = "models"
OUT_PATH = os.path.join(OUT_DIR, "titanic_pipeline.sav")


# FEATURE ENGINEERING

def add_features(df):
    df = df.copy()

    # Extract Title from Name (e.g., Mr, Mrs, Miss)
    df['Title'] = df['Name'].str.extract(r',\s*([^.]*)\.', expand=False).str.strip()

    # Simplify rare titles
    df['Title'] = df['Title'].replace(
        ['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev',
         'Jonkheer', 'Dona', 'Sir', 'the Countess'],
        'Rare'
    )
    df['Title'] = df['Title'].replace(['Mlle', 'Ms'], 'Miss')
    df['Title'] = df['Title'].replace('Mme', 'Mrs')

    # Family size and alone indicator
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    # Extract cabin deck (first letter)
    df['CabinDeck'] = df['Cabin'].fillna('U').astype(str).str[0]

    # Clean missing values for Fare and Age
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())
    df['Age'] = df['Age']  # Will impute later

    return df


# ONE-HOT ENCODER COMPATIBILITY

def make_ohe():
    """Ensures compatibility across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown='ignore', sparse=False)
    except TypeError:
        try:
            return OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown='ignore')



# MAIN TRAINING PIPELINE

def main():
    # Create model output directory
    os.makedirs(OUT_DIR, exist_ok=True)

    # Load dataset
    df = pd.read_csv(DATA_PATH)
    print(f"✅ Loaded dataset: {DATA_PATH}")
    print(f"Shape: {df.shape}\n")

    # Feature engineering
    df = add_features(df)

    # Ensure target column exists
    if 'Survived' not in df.columns:
        raise ValueError("❌ 'Survived' column (target) not found in dataset!")

    # Define features and target
    FEATURES = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked',
        'Title', 'FamilySize', 'IsAlone', 'CabinDeck'
    ]
    TARGET = 'Survived'

    X = df[FEATURES]
    y = df[TARGET].astype(int)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Separate numeric and categorical columns
    num_features = ['Age', 'Fare', 'FamilySize', 'SibSp', 'Parch']
    cat_features = ['Pclass', 'Sex', 'Embarked', 'Title', 'IsAlone', 'CabinDeck']

    # Pipelines for preprocessing
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', make_ohe())
    ])

    preprocessor = ColumnTransformer([
        ('num', num_pipeline, num_features),
        ('cat', cat_pipeline, cat_features)
    ])

    # Complete pipeline with model
    pipeline = Pipeline([
        ('preproc', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=-1
        ))
    ])

    # Train the model
    print("🚀 Training model...")
    pipeline.fit(X_train, y_train)
    print("✅ Training complete.\n")

    # Evaluate model
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]

    print("📊 Evaluation Results:")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    try:
        print("ROC AUC:", roc_auc_score(y_test, y_prob))
    except Exception:
        pass
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # Optional cross-validation
    try:
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1)
        print("\nCV ROC AUC (5-fold): {:.4f} ± {:.4f}".format(cv_scores.mean(), cv_scores.std()))
    except Exception:
        pass

    # Save trained model
    joblib.dump(pipeline, OUT_PATH)
    print(f"\n💾 Model saved successfully at: {OUT_PATH}")

    # Test load
    loaded = joblib.load(OUT_PATH)
    sample = X_test.iloc[[0]]
    print("\n🔍 Sample Input for Verification:")
    print(sample.to_dict(orient='records')[0])
    print("\nPredicted Probability of Survival:", loaded.predict_proba(sample)[:, 1][0])
    print("Predicted Class:", loaded.predict(sample)[0])

    # Drop target column for future prediction dataset
    df_features_only = df.drop(columns=[TARGET])
    df_features_only.to_csv(
        os.path.join("data", "Titanic-Dataset-features-only.csv"),
        index=False
    )
    print("\n🧹 Saved feature-only dataset (without 'Survived') for future predictions.")



# RUN SCRIPT

if __name__ == "__main__":
    main()
