# salary_predictor.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import joblib
import time
import base64
import io
import os 



# Set page config
st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# App header
st.title("💰 Employee Salary Prediction")
st.markdown("""
This app predicts employee salaries based on their profile information.
""")

# Sidebar
st.sidebar.header("About")
st.sidebar.info("""
This application is designed to predict employee salaries using machine learning. 
Upload your dataset or use the default one to train models and make predictions.
""")

# Function to load data
@st.cache_data
def load_data():
    # Create synthetic data if no file is uploaded
    data = {
        'Age': np.random.randint(22, 60, 1000),
        'Gender': np.random.choice(['Male', 'Female'], 1000),
        'Education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000, p=[0.2, 0.5, 0.2, 0.1]),
        'Job_Title': np.random.choice(['Software Engineer', 'Data Scientist', 'Product Manager', 
                                     'HR Manager', 'Marketing Specialist', 'Sales Executive'], 1000),
        'Years_of_Experience': np.random.randint(1, 30, 1000),
        'Department': np.random.choice(['IT', 'HR', 'Finance', 'Marketing', 'Sales'], 1000),
        'Location': np.random.choice(['New York', 'San Francisco', 'Chicago', 'Austin', 'Remote'], 1000),
        'Salary': np.random.randint(50000, 200000, 1000) + 
                 np.random.randint(1, 30, 1000) * 3000  # Salary increases with experience
    }
    return pd.DataFrame(data)

# File upload
uploaded_file = st.file_uploader("Upload your CSV file (optional)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully!")
else:
    df = load_data()
    st.info("Using default synthetic dataset. You can upload your own CSV file.")

# Display data
st.subheader("Dataset Preview")
st.write(df.head())

# EDA Section
st.header("🔍 Exploratory Data Analysis (EDA)")

if st.checkbox("Show Data Summary"):
    st.subheader("Data Summary")
    st.write(df.describe())

if st.checkbox("Show Data Information"):
    st.subheader("Data Information")
    buffer = io.StringIO()
    df.info(buf=buffer)
    st.text(buffer.getvalue())

if st.checkbox("Show Missing Values"):
    st.subheader("Missing Values")
    st.write(df.isnull().sum())

# Visualizations
st.subheader("Data Visualizations")

# Plot selection
plot_type = st.selectbox("Select Plot Type", 
                         ["Histogram", "Box Plot", "Scatter Plot", "Correlation Heatmap", "Count Plot"])

if plot_type == "Histogram":
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_columns) == 0:
        st.warning("No numeric columns available for histogram.")
    else:
        column = st.selectbox("Select Column", num_columns, key="hist_col")
        if column:
            fig, ax = plt.subplots()
            sns.histplot(df[column], kde=True)
            st.pyplot(fig)
    
elif plot_type == "Box Plot":
    num_columns = df.select_dtypes(include=['int64', 'float64']).columns
    if len(num_columns) == 0:
        st.warning("No numeric columns available for box plot.")
    else:
        column = st.selectbox("Select Column", num_columns, key="box_col")
        if column:
            fig, ax = plt.subplots()
            sns.boxplot(x=df[column], ax=ax)
            st.pyplot(fig)
    
elif plot_type == "Scatter Plot":
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    x_col = st.selectbox("X Axis", num_cols)
    y_col = st.selectbox("Y Axis", num_cols, index=len(num_cols)-1)
    hue_col = st.selectbox("Hue (optional)", [None] + list(df.select_dtypes(include=['object']).columns))
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax)
    st.pyplot(fig)
    
elif plot_type == "Correlation Heatmap":
    num_df = df.select_dtypes(include=['int64', 'float64'])
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(num_df.corr(), annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)
    
elif plot_type == "Count Plot":
    column = st.selectbox("Select Column", df.select_dtypes(include=['object']).columns)
    fig, ax = plt.subplots()
    sns.countplot(data=df, y=column, ax=ax, order=df[column].value_counts().index)
    st.pyplot(fig)

# Feature Engineering
st.header("⚙️ Feature Engineering")

# Select target variable
target_col = st.selectbox("Select Target Variable", df.columns, index=len(df.columns)-1)

# Select features
features = st.multiselect("Select Features", df.columns.drop(target_col), default=list(df.columns.drop(target_col)))

# Handle categorical features
cat_cols = df[features].select_dtypes(include=['object']).columns.tolist()
num_cols = df[features].select_dtypes(include=['int64', 'float64']).columns.tolist()

st.subheader("Preprocessing Steps")

# Numeric preprocessing
st.write("**Numeric Features:**", num_cols)
num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical preprocessing
st.write("**Categorical Features:**", cat_cols)
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Combine preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

# Model Selection
st.header("🤖 Model Selection")

# Split data
X = df[features]
y = df[target_col]

test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2, 0.05)
random_state = st.number_input("Random State", 0, 100, 42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Model selection
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(),
    'Lasso Regression': Lasso(),
    'Random Forest': RandomForestRegressor(random_state=random_state),
    'Gradient Boosting': GradientBoostingRegressor(random_state=random_state),
    'Decision Tree': DecisionTreeRegressor(random_state=random_state),
    'SVR': SVR(),
    'KNN': KNeighborsRegressor()
}

selected_models = st.multiselect("Select Models to Compare", list(models.keys()), 
                                default=['Linear Regression', 'Random Forest', 'Gradient Boosting'])

# Model training and evaluation
if st.button("Train Models"):
    results = []
    best_model = None
    best_r2 = -float('inf')
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, (name, model) in enumerate([(name, models[name]) for name in selected_models]):
        status_text.text(f"Training {name}...")
        
        # Create pipeline
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', model)
        ])
        
        # Train model
        start_time = time.time()
        pipeline.fit(X_train, y_train)
        train_time = time.time() - start_time
        
        # Predictions
        y_pred = pipeline.predict(X_test)
        
        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Store results
        results.append({
            'Model': name,
            'MSE': mse,
            'R2 Score': r2,
            'Training Time (s)': train_time
        })
        
        # Check if best model
        if r2 > best_r2:
            best_r2 = r2
            best_model = pipeline
        
        progress_bar.progress((i + 1) / len(selected_models))
    
    # Display results
    results_df = pd.DataFrame(results).sort_values('R2 Score', ascending=False)
    st.subheader("Model Comparison")
    st.table(results_df.style.background_gradient(cmap='Blues', subset=['R2 Score']))
    
    # Plot feature importance for tree-based models
    if 'Random Forest' in selected_models or 'Gradient Boosting' in selected_models or 'Decision Tree' in selected_models:
        st.subheader("Feature Importance")
        
        for name in selected_models:
            if name in ['Random Forest', 'Gradient Boosting', 'Decision Tree']:
                pipeline = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', models[name])
                ])
                pipeline.fit(X_train, y_train)
                
                # Get feature names after one-hot encoding
                try:
                    feature_names = num_cols.copy()
                    for col in cat_cols:
                        categories = pipeline.named_steps['preprocessor'].named_transformers_['cat'].named_steps['onehot'].categories_[cat_cols.index(col)]
                        feature_names.extend([f"{col}_{cat}" for cat in categories])
                    
                    # Get feature importances
                    importances = pipeline.named_steps['regressor'].feature_importances_
                    
                    # Create DataFrame
                    importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
                    importance_df = importance_df.sort_values('Importance', ascending=False).head(10)
                    
                    # Plot
                    fig, ax = plt.subplots()
                    sns.barplot(data=importance_df, x='Importance', y='Feature', ax=ax)
                    ax.set_title(f"Feature Importance - {name}")
                    st.pyplot(fig)
                except Exception as e:
                    st.warning(f"Could not plot feature importance for {name}: {str(e)}")
    
    # Save best model
    if best_model is not None:
        joblib.dump(best_model, 'best_model.pkl')
        st.success(f"Best model saved: {results_df.iloc[0]['Model']} with R2 score of {results_df.iloc[0]['R2 Score']:.3f}")

# Prediction Section
st.header("🔮 Make Predictions")

if st.checkbox("Show Prediction Interface") or 'best_model.pkl' in os.listdir():
    try:
        model = joblib.load('best_model.pkl')
        st.success("Model loaded successfully!")
        
        st.subheader("Enter Employee Details")
        
        # Create input form
        input_data = {}
        col1, col2 = st.columns(2)
        
        for i, feature in enumerate(features):
            col = col1 if i % 2 == 0 else col2
            
            if feature in num_cols:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                default_val = float(df[feature].median())
                input_data[feature] = col.number_input(
                    f"{feature}", min_val, max_val, default_val
                )
            elif feature in cat_cols:
                options = df[feature].unique().tolist()
                input_data[feature] = col.selectbox(
                    f"{feature}", options
                )
        
        if st.button("Predict Salary"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            st.success(f"Predicted Salary: ${prediction:,.2f}")
            
            # Show confidence interval (simplified)
            st.info(f"Estimated range: ${prediction*0.9:,.2f} - ${prediction*1.1:,.2f}")
            
    except Exception as e:
        st.error(f"Error loading model: {str(e)}. Please train a model first.")

# Download section
st.header("📥 Download")

if st.button("Download Sample CSV Template"):
    sample_df = df.head(10).drop(columns=[target_col])
    csv = sample_df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="salary_template.csv">Download CSV Template</a>'
    st.markdown(href, unsafe_allow_html=True)

if 'best_model.pkl' in os.listdir():
    with open('best_model.pkl', 'rb') as f:
        st.download_button(
            label="Download Trained Model",
            data=f,
            file_name="salary_predictor_model.pkl",
            mime="application/octet-stream"
        )

# Footer
st.markdown("---")
st.markdown("""
**Employee Salary Prediction App**  
Created with Streamlit and Scikit-learn
""")







# End of the app
if __name__ == "__main__":
    st.write("Run this app using `streamlit run salary_predictor.py`")