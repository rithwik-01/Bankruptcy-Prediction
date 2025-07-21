"""
Bankruptcy Prediction using Machine Learning
Dataset: Polish Companies Bankruptcy Data
"""

import zipfile
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.io import arff
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE

# Optional: Install required packages (uncomment if needed)
# pip install xgboost imbalanced-learn

try:
    from xgboost import XGBClassifier
    from sklearn.svm import SVC
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.calibration import CalibratedClassifierCV
    ADVANCED_MODELS_AVAILABLE = True
except ImportError:
    print("Warning: Some advanced models (XGBoost) may not be available.")
    ADVANCED_MODELS_AVAILABLE = False

# =============================================================================
# CONFIGURATION
# =============================================================================

# Define file paths
ZIP_FILE_PATH = r"D:\Bankruptcy-Prediction\polish+companies+bankruptcy+data.zip"
EXTRACTION_PATH = r"D:\Bankruptcy-Prediction\extracted_data"

# Feature names mapping dictionary
FEATURE_NAMES_DICT = {
    "X1": "Net Profit / Total Assets",
    "X2": "Total Liabilities / Total Assets",
    "X3": "Working Capital / Total Assets",
    "X4": "Current Assets / Short-Term Liabilities",
    "X5": "Cash Ratio",
    "X6": "Retained Earnings / Total Assets",
    "X7": "EBIT / Total Assets",
    "X8": "Book Value of Equity / Total Liabilities",
    "X9": "Sales / Total Assets",
    "X10": "Equity / Total Assets",
    "X11": "Gross Profit + Extraordinary Items / Total Assets",
    "X12": "Gross Profit / Short-Term Liabilities",
    "X13": "Gross Profit + Depreciation / Sales",
    "X14": "Gross Profit + Interest / Total Assets",
    "X15": "Debt Repayment Ratio",
    "X16": "Gross Profit + Depreciation / Total Liabilities",
    "X17": "Total Assets / Total Liabilities",
    "X18": "Gross Profit / Total Assets",
    "X19": "Gross Profit / Sales",
    "X20": "Inventory Turnover",
    "X21": "Sales (n) / Sales (n-1)",
    "X22": "Profit on Operating Activities / Total Assets",
    "X23": "Net Profit / Sales",
    "X24": "Gross Profit (3 years) / Total Assets",
    "X25": "Equity - Share Capital / Total Assets",
    "X26": "Net Profit + Depreciation / Total Liabilities",
    "X27": "Profit on Operating Activities / Financial Expenses",
    "X28": "Working Capital / Fixed Assets",
    "X29": "Logarithm of Total Assets",
    "X30": "Total Liabilities - Cash / Sales",
    "X31": "Gross Profit + Interest / Sales",
    "X32": "Current Liabilities Turnover",
    "X33": "Operating Expenses / Short-Term Liabilities",
    "X34": "Profit on Sales / Total Assets",
    "X35": "Total Sales / Total Assets",
    "X36": "Current Assets - Inventories / Long-Term Liabilities",
    "X37": "Constant Capital / Total Assets",
    "X38": "Profit on Sales / Sales",
    "X39": "Receivables Turnover Ratio",
    "X40": "Liquidity Ratio",
    "X41": "Debt Coverage Ratio",
    "X42": "Rotation Receivables + Inventory Turnover",
    "X43": "Receivables * 365 / Sales",
    "X44": "Net Profit / Inventory",
    "X45": "EBITDA / Total Assets",
    "X46": "EBITDA / Sales",
    "X47": "Current Assets / Total Liabilities",
    "X48": "Short-Term Liabilities / Total Assets",
    "X49": "Short-Term Liabilities * 365 / Cost of Products Sold",
    "X50": "Equity / Fixed Assets",
    "X51": "Constant Capital / Fixed Assets",
    "X52": "Working Capital",
    "X53": "Sales - Cost of Products Sold / Sales",
    "X54": "Short-Term Liabilities / Sales",
    "X55": "Long-Term Liabilities / Equity",
    "X56": "Sales / Inventory",
    "X57": "Sales / Receivables",
    "X58": "Sales / Short-Term Liabilities",
    "X59": "Sales / Fixed Assets"
}

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

def extract_data():
    """Extract the ZIP file containing the bankruptcy data."""
    if not os.path.exists(ZIP_FILE_PATH):
        raise FileNotFoundError(f"ZIP file not found at: {ZIP_FILE_PATH}")
    
    # Create the extraction directory
    os.makedirs(EXTRACTION_PATH, exist_ok=True)
    
    # Extract the ZIP file
    with zipfile.ZipFile(ZIP_FILE_PATH, 'r') as zip_ref:
        zip_ref.extractall(EXTRACTION_PATH)
    
    print(f"Files extracted to: {EXTRACTION_PATH}")
    print("Extracted files:", os.listdir(EXTRACTION_PATH))

def load_arff_to_dataframe(arff_file_path):
    """
    Convert a .arff file into a Pandas DataFrame.
    
    Args:
        arff_file_path (str): Path to the .arff file.
        
    Returns:
        pd.DataFrame: Loaded data as a Pandas DataFrame.
    """
    if not os.path.exists(arff_file_path):
        raise FileNotFoundError(f"ARFF file not found: {arff_file_path}")
    
    data, meta = arff.loadarff(arff_file_path)
    df = pd.DataFrame(data)
    
    # Handle byte strings in the target column
    for col in df.columns:
        if df[col].dtype == 'object':
            # Convert byte strings to regular strings, then to appropriate type
            df[col] = df[col].astype(str).str.replace("b'", "").str.replace("'", "")
            # Try to convert to numeric if possible
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
    
    return df

def load_all_data():
    """Load all yearly datasets and combine them."""
    dataframes = {}
    
    # Load all .arff files (1year.arff to 5year.arff)
    for i in range(1, 6):
        arff_file_path = os.path.join(EXTRACTION_PATH, f'{i}year.arff')
        try:
            df = load_arff_to_dataframe(arff_file_path)
            dataframes[f'{i}year'] = df
            print(f"Loaded {i}year.arff: {df.shape}")
        except FileNotFoundError:
            print(f"Warning: {i}year.arff not found, skipping...")
    
    if not dataframes:
        raise ValueError("No data files were loaded successfully.")
    
    # Combine all DataFrames
    combined_df = pd.concat(list(dataframes.values()), ignore_index=True)
    print(f"Combined dataset shape: {combined_df.shape}")
    
    return combined_df, dataframes

def preprocess_data(df):
    """Preprocess the combined dataset."""
    print("\n=== DATA PREPROCESSING ===")
    
    # Display basic information
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"\nColumns with missing values:")
    print(missing_values[missing_values > 0])
    
    # Calculate percentage of data loss if rows with missing values are dropped
    total_rows = df.shape[0]
    rows_with_nan = df.dropna().shape[0]
    percentage_lost = ((total_rows - rows_with_nan) / total_rows) * 100
    print(f"Percentage of data lost by dropping rows with missing values: {percentage_lost:.2f}%")
    
    return df

def split_features_target(df, target_column='class'):
    """Split dataset into features and target variable."""
    if target_column not in df.columns:
        # Try to find the target column
        possible_targets = [col for col in df.columns if 'class' in col.lower()]
        if possible_targets:
            target_column = possible_targets[0]
            print(f"Using '{target_column}' as target column")
        else:
            raise ValueError(f"Target column '{target_column}' not found in dataset")
    
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    
    # Ensure target is numeric
    if y.dtype == 'object':
        y = pd.to_numeric(y, errors='coerce')
    
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Target classes: {y.unique()}")
    print(f"Class distribution:")
    print(y.value_counts())
    
    return X, y

def handle_missing_values(X_train, X_test, threshold=0.05):
    """Handle missing values by dropping columns with too many missing values and imputing the rest."""
    print("\n=== HANDLING MISSING VALUES ===")
    
    # Identify columns with missing values
    missing_values = X_train.isnull().sum()
    print("Columns with missing values:")
    print(missing_values[missing_values > 0])
    
    # Determine which columns to drop vs impute
    threshold_count = threshold * X_train.shape[0]
    cols_to_drop = [col for col in X_train.columns if X_train[col].isnull().sum() > threshold_count]
    cols_to_impute = [col for col in X_train.columns if X_train[col].isnull().sum() <= threshold_count and X_train[col].isnull().sum() > 0]
    
    print(f"Columns to drop (>{threshold*100}% missing): {len(cols_to_drop)}")
    print(f"Columns to impute: {len(cols_to_impute)}")
    
    # Drop columns with excessive missing values
    if cols_to_drop:
        X_train = X_train.drop(columns=cols_to_drop)
        X_test = X_test.drop(columns=cols_to_drop)
    
    # Impute remaining missing values using median strategy
    if cols_to_impute:
        imputer = SimpleImputer(strategy='median')
        X_train_imputed = imputer.fit_transform(X_train)
        X_test_imputed = imputer.transform(X_test)
        
        # Convert back to DataFrame
        X_train = pd.DataFrame(X_train_imputed, columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(X_test_imputed, columns=X_test.columns, index=X_test.index)
    
    print(f"Final shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Missing values remaining - Train: {X_train.isnull().sum().sum()}, Test: {X_test.isnull().sum().sum()}")
    
    return X_train, X_test

def scale_features(X_train, X_test):
    """Standardize features using StandardScaler."""
    print("\n=== FEATURE SCALING ===")
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrames
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    print("Feature scaling completed")
    print(f"Train set mean: {X_train_scaled.mean().mean():.6f}")
    print(f"Train set std: {X_train_scaled.std().mean():.6f}")
    
    return X_train_scaled, X_test_scaled, scaler

def apply_smote(X_train, y_train):
    """Apply SMOTE for handling class imbalance."""
    print("\n=== APPLYING SMOTE ===")
    
    print("Class distribution before SMOTE:")
    print(y_train.value_counts())
    
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    print("\nClass distribution after SMOTE:")
    print(pd.Series(y_train_smote).value_counts())
    
    return X_train_smote, y_train_smote

# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def plot_class_distribution(y_before, y_after):
    """Plot class distribution before and after SMOTE."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before SMOTE
    sns.countplot(x=y_before, palette="viridis", ax=ax1)
    ax1.set_title("Class Distribution Before SMOTE")
    ax1.set_xlabel("Class")
    ax1.set_ylabel("Count")
    
    # After SMOTE
    sns.countplot(x=y_after, palette="viridis", ax=ax2)
    ax2.set_title("Class Distribution After SMOTE")
    ax2.set_xlabel("Class")
    ax2.set_ylabel("Count")
    
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, top_n=10):
    """Plot feature importance for tree-based models."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Map feature names to readable names
        readable_names = [FEATURE_NAMES_DICT.get(name, name) for name in feature_names]
        
        # Select top N features
        top_features = [readable_names[i] for i in indices[:top_n]]
        top_importances = importances[indices[:top_n]]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(top_n), top_importances[::-1], color='skyblue')
        plt.yticks(range(top_n), top_features[::-1])
        plt.xlabel("Feature Importance")
        plt.title(f"Top {top_n} Features by Importance")
        plt.tight_layout()
        plt.show()

def plot_confusion_matrices(models_results):
    """Plot confusion matrices for multiple models."""
    n_models = len(models_results)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (name, results) in enumerate(models_results.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        cm = confusion_matrix(results['y_test'], results['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title(f"{name} Confusion Matrix")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    
    # Hide empty subplots
    for idx in range(n_models, rows * cols):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()

# =============================================================================
# MODEL TRAINING AND EVALUATION
# =============================================================================

def train_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """Train and evaluate a single model."""
    print(f"\n=== {model_name.upper()} ===")
    
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
    print(f"Confusion Matrix:\n{cm}")
    
    return {
        'model': model,
        'y_pred': y_pred,
        'y_test': y_test,
        'accuracy': accuracy,
        'report': report,
        'confusion_matrix': cm
    }

def compare_models(results_dict):
    """Create a comparison table for all models."""
    comparison_data = []
    
    for name, results in results_dict.items():
        try:
            precision = results['report']['1']['precision']
            recall = results['report']['1']['recall']
            f1_score = results['report']['1']['f1-score']
        except KeyError:
            # Handle case where class '1' doesn't exist (use macro average)
            precision = results['report']['macro avg']['precision']
            recall = results['report']['macro avg']['recall']
            f1_score = results['report']['macro avg']['f1-score']
        
        comparison_data.append({
            'Model': name,
            'Accuracy': results['accuracy'],
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n=== MODEL COMPARISON ===")
    print(comparison_df.round(4))
    
    return comparison_df

# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    try:
        # Step 1: Extract data
        print("=== EXTRACTING DATA ===")
        extract_data()
        
        # Step 2: Load data
        print("\n=== LOADING DATA ===")
        combined_df, individual_dfs = load_all_data()
        
        # Step 3: Preprocess data
        combined_df = preprocess_data(combined_df)
        
        # Step 4: Split features and target
        X, y = split_features_target(combined_df)
        
        # Step 5: Split into train/test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        print(f"\nTrain set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Step 6: Handle missing values
        X_train, X_test = handle_missing_values(X_train, X_test)
        
        # Step 7: Scale features
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
        
        # Step 8: Apply SMOTE (optional)
        X_train_smote, y_train_smote = apply_smote(X_train_scaled, y_train)
        
        # Plot class distribution
        plot_class_distribution(y_train, y_train_smote)
        
        # Step 9: Train and evaluate models
        models_results = {}
        
        # Basic models
        models = {
            'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
            'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        }
        
        # Add advanced models if available
        if ADVANCED_MODELS_AVAILABLE:
            models.update({
                'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
                'SVM': SVC(kernel='rbf', C=1, gamma='scale', probability=True, random_state=42),
                'KNN': KNeighborsClassifier(n_neighbors=5)
            })
        
        # Train all models
        for name, model in models.items():
            results = train_evaluate_model(
                model, name, X_train_scaled, X_test_scaled, y_train, y_test
            )
            models_results[name] = results
        
        # Step 10: Compare models
        comparison_df = compare_models(models_results)
        
        # Step 11: Visualizations
        # Feature importance for Random Forest
        if 'Random Forest' in models_results:
            plot_feature_importance(
                models_results['Random Forest']['model'], 
                X_train.columns, 
                top_n=10
            )
        
        # Confusion matrices
        plot_confusion_matrices(models_results)
        
        print("\n=== ANALYSIS COMPLETE ===")
        return models_results, comparison_df
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    results, comparison = main()