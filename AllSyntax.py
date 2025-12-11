# ==============================================================================
# MASTER SYNTAX REFERENCE (syntax.py)
# ==============================================================================
# USAGE: 
# 1. Copy the imports you need.
# 2. Copy specific model definitions from Section 2.
# 3. Use Section 3 as a template for your main project structure.
# ==============================================================================

# --- ESSENTIAL IMPORTS ---
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- SCIKIT-LEARN IMPORTS ---
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score

# --- MODEL IMPORTS ---
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# --- XGBOOST (Requires: pip install xgboost) ---
try:
    from xgboost import XGBClassifier, XGBRegressor
except ImportError:
    print("XGBoost not installed. Run: pip install xgboost")


# ==============================================================================
# SECTION 1: PREPROCESSING & HEALTH CHECK
# ==============================================================================
def check_data_health(df):
    """
    Run this function to inspect a new dataframe for missing values, 
    outliers, and data types.
    """
    print("================ DATA HEALTH CHECK ================")
    
    # 1. INFO & TYPES
    print("\n--- 1. INFO (Nulls & Types) ---")
    print(df.info())
    # INTERPRETATION:
    # - If 'Age' has 700 non-null but total entries are 891 -> You must IMPUTE.
    # - If 'Price' is 'object', it likely contains symbols ($) -> Needs cleaning.

    # 2. STATISTICS
    print("\n--- 2. STATISTICS (Scale & Outliers) ---")
    print(df.describe())
    # INTERPRETATION:
    # - Compare Mean vs 50% (Median). Large gap = Skewed data.
    # - Compare Max vs 75%. If Max is huge (e.g., 500) vs 75% (30) -> OUTLIERS.

    # 3. MISSING VALUES
    print("\n--- 3. MISSING VALUE COUNT ---")
    print(df.isnull().sum())
    
    # Visual Heatmap
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Data Heatmap (Yellow = Missing)")
    plt.show()

    # 4. CATEGORICAL SCAN
    print("\n--- 4. CATEGORICAL COLUMNS ---")
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in cat_cols:
        print(f"\nColumn: {col}")
        print(df[col].value_counts().head(5))
    # INTERPRETATION:
    # - Low Cardinality (e.g., "Red", "Blue") -> Use OneHotEncoder.
    # - High Cardinality (e.g., 500 Cities) -> Use LabelEncoder or grouping.

    # 5. CORRELATION MATRIX (Numeric Only)
    print("\n--- 5. CORRELATION MATRIX ---")
    plt.figure(figsize=(10,8))
    sns.heatmap(df.select_dtypes(include='number').corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.show()
    # INTERPRETATION:
    # - Dark Red (>0.90): Multicollinearity. Drop one of the duplicates.

# ==============================================================================
# CRITICAL: ORDINAL VS. NOMINAL DATA (THE "0.88 to 0.98" TRAP)
# ==============================================================================

def check_ordinality(df, column_name):
    """
    BEFORE encoding, ask yourself: "Does the order matter?"
    
    SCENARIO A: NOMINAL (Order DOES NOT matter)
    Examples: "City" (Paris, London), "Color" (Red, Blue), "Type" (Sedan, SUV)
    - Logic: Paris is not "greater than" London.
    - Action: Use OneHotEncoder.
    - Code: pd.get_dummies(df, columns=['City'])
    
    SCENARIO B: ORDINAL (Order MATTERS)
    Examples: "Size" (S, M, L), "Rating" (Bad, Good, Excellent), "Quality" (Low, High)
    - Logic: L > M > S. "Excellent" > "Good".
    - Trap: If you use OneHotEncoder here, the model loses the ranking information.
            It thinks "Large" and "Small" are just different categories, not ranks.
            Your score will drop (e.g., from 0.98 to 0.88).
    
    - Action: YOU MUST MAP IT MANUALLY OR USE ORDINAL ENCODER.
    """
    
    # EXAMPLE CODE FOR ORDINAL MAPPING
    # 1. Define the specific rank (Human Logic Required)
    rank_mapping = {
        'Low': 1,
        'Medium': 2,
        'High': 3,
        'Ultra': 4
    }
    
    # 2. Apply the map
    # df[column_name] = df[column_name].map(rank_mapping)
    
    print(f"Applied ordinal mapping to {column_name}")
# ==============================================================================
# SECTION 2: MODEL ZOO (COPY-PASTE READY)
# ==============================================================================
def get_model_syntax():
    """
    A reference dictionary for model initialization and hyperparameters.
    Do not run this function; copy the code inside.
    """
    models = {}

    # 1. LINEAR REGRESSION
    # Use for: Predicting numbers (House Price, Temperature). Baseline.
    models['LinearRegression'] = LinearRegression()

    # 2. LOGISTIC REGRESSION
    # Use for: Binary Classification (Spam/Not Spam).
    # Params: C (Inverse regularization). High C = Overfit risk. Low C = Underfit.
    models['LogisticRegression'] = LogisticRegression(C=1.0, max_iter=1000)

    # 3. DECISION TREES
    # Use for: Interpretability (Visualizing decisions).
    # Params: max_depth (Limits complexity).
    models['Tree_Classifier'] = DecisionTreeClassifier(max_depth=5, min_samples_split=10)
    models['Tree_Regressor']  = DecisionTreeRegressor(max_depth=5)

    # 4. RANDOM FOREST (The Reliable Workhorse)
    # Use for: High accuracy, general purpose.
    # Params: n_estimators (Trees), n_jobs=-1 (Use all CPU cores).
    models['Forest_Regressor'] = RandomForestRegressor(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
    )
    models['Forest_Classifier'] = RandomForestClassifier(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=42
    )

    # 5. XGBOOST (The Kaggle Winner)
    # Use for: Max performance on tabular data.
    # Params: learning_rate (eta), subsample (prevents overfit).
    models['XGB_Regressor'] = XGBRegressor(
        n_estimators=1000, learning_rate=0.05, max_depth=5, 
        subsample=0.8, n_jobs=-1
    )

    # 6. SUPPORT VECTOR MACHINES (SVM)
    # Use for: Complex boundaries, small datasets.
    # Params: kernel ('rbf' for curves), C, gamma.
    models['SVM_Classifier'] = SVC(kernel='rbf', C=1.0, gamma='scale')

    # 7. KNN (K-NEAREST NEIGHBORS)
    # Use for: Similarity grouping. NOTE: SCALING IS MANDATORY.
    # Params: n_neighbors (Low=Jagged/Overfit, High=Smooth/Underfit).
    models['KNN'] = KNeighborsClassifier(n_neighbors=5)

    return models


# ==============================================================================
# SECTION 3: THE "PROFESSIONAL" PIPELINE TEMPLATE
# ==============================================================================
def run_full_workflow_example(df, target_col):
    """
    A template for the full end-to-end workflow: 
    Split -> Preprocess -> Pipeline -> GridSearch -> Evaluate
    """
    
    # ------------------------------------------
    # STEP 1: DEFINE FEATURES AND TARGET
    # ------------------------------------------
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    # ------------------------------------------
    # STEP 2: SPLIT DATA
    # ------------------------------------------
    # Always split BEFORE scaling to avoid data leakage
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ------------------------------------------
    # STEP 3: PREPROCESSING PIPELINE
    # ------------------------------------------
    # Identify column types
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object', 'category']).columns

    # Recipe for Numbers: Fill Missing -> Scale
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])

    # Recipe for Text: Fill Missing -> OneHotEncode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Bundle recipes
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    # ------------------------------------------
    # STEP 4: DEFINE MODEL PIPELINE
    # ------------------------------------------
    # Pipeline = Preprocessor + Model
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(random_state=42)) 
    ])

    # ------------------------------------------
    # STEP 5: HYPERPARAMETER TUNING (GridSearch)
    # ------------------------------------------
    # Dictionary of parameters to test
    param_grid = {
        'regressor__n_estimators': [100, 200],
        'regressor__max_depth': [10, 20, None],
        'regressor__min_samples_split': [2, 5]
    }

    print("Training and tuning... this might take a moment.")
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, scoring='r2', n_jobs=-1)
    
    # FIT THE MODEL (The heavy lifting happens here)
    grid_search.fit(X_train, y_train)

    # ------------------------------------------
    # STEP 6: EVALUATE
    # ------------------------------------------
    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Training Score: {grid_search.best_score_:.4f}")

    # Final Exam: Predict on Test Data
    y_pred = grid_search.predict(X_test)
    final_score = r2_score(y_test, y_pred)

    print(f"Final Test R2 Score: {final_score:.4f}")
    
    return grid_search # Return the best model