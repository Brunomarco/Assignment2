import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
    classification_report, mean_squared_error, r2_score
)
import joblib
import io
# Page configuration
st.set_page_config(page_title="ML Model Trainer", layout="wide")
st.title("Interactive ML Model Trainer")

# Initialize session state
if 'model_trained' not in st.session_state:
    st.session_state.model_trained = False
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'pipeline' not in st.session_state:
    st.session_state.pipeline = None
if 'metrics' not in st.session_state:
    st.session_state.metrics = {}
if 'target_is_categorical' not in st.session_state:
    st.session_state.target_is_categorical = None

# Load datasets with caching
@st.cache_data
def load_datasets():
    return {
        'Titanic': sns.load_dataset('titanic'),
        'Tips': sns.load_dataset('tips'),
        'Penguins': sns.load_dataset('penguins'),
        'Iris': sns.load_dataset('iris'),
        'Diamonds': sns.load_dataset('diamonds').sample(1000)
    }

# Create tabs
tab1, tab2, tab3 = st.tabs(["📊 Dataset & Features", "🔧 Model Configuration", "📈 Results"])

# Dataset Selection Tab
with tab1:
    st.header("Dataset Selection")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        data_source = st.radio("Data Source", ["Sample Datasets", "Upload Your Own"])
    
    with col2:
        if data_source == "Sample Datasets":
            datasets = load_datasets()
            dataset_name = st.selectbox("Select Dataset", list(datasets.keys()))
            df = datasets[dataset_name]
        else:
            uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file)
                    st.success("✅ Custom dataset loaded")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    datasets = load_datasets()
                    df = datasets["Titanic"]  # Default
            else:
                st.warning("Please upload a CSV file")
                datasets = load_datasets()
                df = datasets["Titanic"]  # Default
    
    # Check if dataset changed
    current_data_hash = hash(df.to_string())
    if st.session_state.current_dataset != current_data_hash:
        st.session_state.current_dataset = current_data_hash
        st.session_state.model_trained = False
    
    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)
    st.write(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    
    # Feature selection
    st.subheader("🎯 Feature & Target Selection")
    col1, col2 = st.columns([1, 1])
    
    all_columns = df.columns.tolist()
    
    with col1:
        target = st.selectbox("Target Variable", all_columns)
        
        # Detect target type
        target_is_categorical = df[target].dtype == 'object' or df[target].nunique() <= 10
        st.session_state.target_is_categorical = target_is_categorical
        
        target_type = "categorical" if target_is_categorical else "numerical"
        st.info(f"Target '{target}' detected as {target_type}")
    
    with col2:
        # Separate numerical and categorical features
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        
        # Remove target from potential features
        if target in numerical_cols:
            numerical_cols.remove(target)
        if target in categorical_cols:
            categorical_cols.remove(target)
        
        num_features = st.multiselect("Numerical Features", numerical_cols, default=numerical_cols)
        cat_features = st.multiselect("Categorical Features", categorical_cols, default=categorical_cols)
        
        features = num_features + cat_features
        
        if not features:
            st.warning("Please select at least one feature")

# Model Configuration Tab
with tab2:
    st.header("Model Configuration")
    
    # Only enable if features are selected
    if 'features' in locals() and features:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Data Splitting")
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
            random_state = st.slider("Random State", 0, 100, 42)
            
            st.subheader("Model Selection")
            if st.session_state.target_is_categorical:
                model_type = "Classification"
                model_options = ["Logistic Regression", "Random Forest"]
            else:
                model_type = "Regression"
                model_options = ["Linear Regression", "Random Forest"]
            
            model_name = st.selectbox(f"Choose {model_type} Model", model_options)
        
        with col2:
            st.subheader("Model Parameters")
            
            if model_name == "Logistic Regression":
                C = st.slider("Regularization strength (C)", 0.01, 10.0, 1.0)
                max_iter = st.slider("Maximum iterations", 100, 1000, 500, 100)
                model = LogisticRegression(C=C, max_iter=max_iter, random_state=random_state)
                
            elif model_name == "Linear Regression":
                fit_intercept = st.checkbox("Fit intercept", True)
                model = LinearRegression(fit_intercept=fit_intercept) 
                
            elif model_name == "Random Forest":
                n_estimators = st.slider("Number of trees", 10, 200, 100, 10)
                max_depth = st.slider("Maximum depth", 1, 20, 5)
                
                if st.session_state.target_is_categorical:
                    model = RandomForestClassifier(
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        random_state=random_state
                    )
                else:
                    model = RandomForestRegressor(
                        n_estimators=n_estimators, 
                        max_depth=max_depth,
                        random_state=random_state
                    )
            
            # Preprocessing options
            st.subheader("Preprocessing")
            scaling = st.checkbox("Apply Standard Scaling", True)
            impute = st.checkbox("Impute Missing Values", True)
        
        # Train model button
        with st.form("Train Form"):
            st.subheader("🚀 Train Model")
            submitted = st.form_submit_button("👟 Fit Model")
        
        if submitted:
            # Prepare data
            X = df[features].copy()
            y = df[target].copy()
            
            # Encode categorical target if needed
            if st.session_state.target_is_categorical:
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            
            # Show progress
            with st.spinner('Training model...'):
                # Set up preprocessing
                numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
                categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
                
                preprocessing_steps = []
                
                # Numeric preprocessing
                if numerical_features:
                    numeric_pipeline_steps = []
                    
                    if impute:
                        numeric_pipeline_steps.append(('imputer', SimpleImputer(strategy="mean")))
                    
                    if scaling:
                        numeric_pipeline_steps.append(('scaler', StandardScaler()))
                    
                    if numeric_pipeline_steps:
                        preprocessing_steps.append(('num', Pipeline(steps=numeric_pipeline_steps), numerical_features))
                
                # Categorical preprocessing
                if categorical_features:
                    cat_pipeline_steps = []
                    
                    if impute:
                        cat_pipeline_steps.append(('imputer', SimpleImputer(strategy='most_frequent')))
                    
                    cat_pipeline_steps.append(('onehot', OneHotEncoder(handle_unknown='ignore')))
                    
                    if cat_pipeline_steps:
                        preprocessing_steps.append(('cat', Pipeline(steps=cat_pipeline_steps), categorical_features))
                
                # Create preprocessing pipeline
                if preprocessing_steps:
                    preprocessor = ColumnTransformer(preprocessing_steps)
                    pipeline = Pipeline([
                        ('preprocess', preprocessor),
                        ('model', model)
                    ])
                else:
                    pipeline = Pipeline([('model', model)])
                
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                if st.session_state.target_is_categorical:
                    # Classification metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
                    recall = recall_score(y_test, y_pred, average='weighted')
                    f1 = f1_score(y_test, y_pred, average='weighted')
                    
                    # Store metrics
                    st.session_state.metrics = {
                        "Accuracy": accuracy,
                        "Precision": precision,
                        "Recall": recall,
                        "F1 Score": f1,
                        "Classification Report": classification_report(y_test, y_pred),
                        "Confusion Matrix": confusion_matrix(y_test, y_pred)
                    }
                    
                    # ROC curve for binary classification
                    if len(np.unique(y)) == 2 and hasattr(pipeline.named_steps['model'], 'predict_proba'):
                        probs = pipeline.predict_proba(X_test)[:, 1]
                        fpr, tpr, _ = roc_curve(y_test, probs)
                        roc_auc = auc(fpr, tpr)
                        st.session_state.metrics["ROC"] = {
                            "fpr": fpr,
                            "tpr": tpr,
                            "auc": roc_auc
                        }
                else:
                    # Regression metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    r2 = r2_score(y_test, y_pred)
                    residuals = y_test - y_pred
                    
                    # Store metrics
                    st.session_state.metrics = {
                        "MSE": mse,
                        "RMSE": rmse,
                        "R2 Score": r2,
                        "Actual": y_test,
                        "Predicted": y_pred,
                        "Residuals": residuals
                    }
                
                # Store model
                st.session_state.pipeline = pipeline
                st.session_state.model_trained = True
                st.session_state.model_name = model_name
                st.session_state.features = features
                st.session_state.target = target
                
                # Show success message
                st.success("Model trained successfully! Go to the Results tab to see evaluation metrics.")
    else:
        st.warning("Please select features and a target variable in the Dataset & Features tab.")

# Results Tab
with tab3:
    st.header("Model Results & Evaluation")
    
    if st.session_state.model_trained:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Model Performance Metrics")
            
            if st.session_state.target_is_categorical:
                # Classification metrics
                metrics_df = pd.DataFrame({
                    "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
                    "Value": [
                        st.session_state.metrics["Accuracy"],
                        st.session_state.metrics["Precision"],
                        st.session_state.metrics["Recall"],
                        st.session_state.metrics["F1 Score"]
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True)
                
                # Confusion Matrix
                st.write("Confusion Matrix")
                fig_cm, ax_cm = plt.subplots(figsize=(6, 4))
                cm = st.session_state.metrics["Confusion Matrix"]
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                ax_cm.set_xlabel('Predicted')
                ax_cm.set_ylabel('Actual')
                st.pyplot(fig_cm)
                
                # ROC Curve for binary classification
                if "ROC" in st.session_state.metrics:
                    st.write("ROC Curve")
                    fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
                    roc_data = st.session_state.metrics["ROC"]
                    ax_roc.plot(roc_data["fpr"], roc_data["tpr"], label=f"AUC = {roc_data['auc']:.2f}")
                    ax_roc.plot([0, 1], [0, 1], linestyle='--', color='gray')
                    ax_roc.set_xlabel("False Positive Rate")
                    ax_roc.set_ylabel("True Positive Rate")
                    ax_roc.legend()
                    st.pyplot(fig_roc)
            else:
                # Regression metrics
                metrics_df = pd.DataFrame({
                    "Metric": ["MSE", "RMSE", "R2 Score"],
                    "Value": [
                        st.session_state.metrics["MSE"],
                        st.session_state.metrics["RMSE"],
                        st.session_state.metrics["R2 Score"]
                    ]
                })
                st.dataframe(metrics_df, use_container_width=True)
                
                # Actual vs Predicted
                st.write("Actual vs Predicted Values")
                actual = st.session_state.metrics["Actual"]
                predicted = st.session_state.metrics["Predicted"]
                
                fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
                ax_scatter.scatter(actual, predicted, alpha=0.5)
                ax_scatter.plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--')
                ax_scatter.set_xlabel('Actual')
                ax_scatter.set_ylabel('Predicted')
                st.pyplot(fig_scatter)
                
                # Residuals
                st.write("Residual Distribution")
                residuals = st.session_state.metrics["Residuals"]
                
                fig_residuals = plt.figure(figsize=(6, 4))
                plt.hist(residuals, bins=20, alpha=0.7)
                plt.axvline(x=0, color='red', linestyle='--')
                plt.xlabel("Residuals")
                plt.ylabel("Frequency")
                st.pyplot(fig_residuals)
        
        with col2:
            st.subheader("📈 Feature Importance")
            
            model = st.session_state.pipeline.named_steps['model']
            
            # Get feature importance if available
            if hasattr(model, 'feature_importances_'):
                # Get feature names from preprocessor
                if 'preprocess' in st.session_state.pipeline.named_steps:
                    feature_names = st.session_state.pipeline.named_steps['preprocess'].get_feature_names_out()
                else:
                    feature_names = st.session_state.features
                
                # Create feature importance dataframe
                importances = model.feature_importances_
                
                # Ensure we have the right number of feature names
                if len(importances) == len(feature_names):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Importance': importances
                    }).sort_values('Importance', ascending=False)
                    
                    # Plot feature importance
                    fig = plt.figure(figsize=(8, 6))
                    plt.barh(importance_df['Feature'][:10], importance_df['Importance'][:10])
                    plt.title("Top 10 Feature Importances")
                    plt.xlabel("Importance")
                    st.pyplot(fig)
            elif hasattr(model, 'coef_'):
                # Linear models
                if 'preprocess' in st.session_state.pipeline.named_steps:
                    feature_names = st.session_state.pipeline.named_steps['preprocess'].get_feature_names_out()
                else:
                    feature_names = st.session_state.features
                
                # Get coefficients
                coefficients = model.coef_
                
                # Ensure we have the right dimensions
                if coefficients.ndim == 1:
                    coef_array = coefficients
                elif coefficients.ndim == 2 and coefficients.shape[0] == 1:
                    coef_array = coefficients[0]
                else:
                    coef_array = np.mean(np.abs(coefficients), axis=0)
                
                # Create importance dataframe
                if len(coef_array) == len(feature_names):
                    importance_df = pd.DataFrame({
                        'Feature': feature_names,
                        'Coefficient': coef_array
                    }).sort_values('Coefficient', ascending=False)
                    
                    # Plot coefficients
                    fig = plt.figure(figsize=(8, 6))
                    plt.barh(importance_df['Feature'][:10], importance_df['Coefficient'][:10])
                    plt.title("Top 10 Feature Coefficients")
                    plt.xlabel("Coefficient Value")
                    st.pyplot(fig)
            
            # Model Export
            st.subheader("💾 Download Model")
            buffer = io.BytesIO()
            joblib.dump(st.session_state.pipeline, buffer)
            st.download_button(
                "Download trained model (.pkl)",
                data=buffer.getvalue(),
                file_name="trained_model.pkl",
                mime="application/octet-stream"
            )
    else:
        st.info("No model trained yet. Go to the Model Configuration tab to train a model.")
