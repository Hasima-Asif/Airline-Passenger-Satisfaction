import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Comprehensive CSS for sea green background and black text
st.markdown("""
    <style>
    .stApp {
        background-color: #28282B;
        font-family: 'Arial', sans-serif;
        color: #FFFFFF !important;
    }

    /* General form elements */
    .markdown-text-container, .stMarkdown, .stText, .stDataFrame, .stTable,
    .stSlider, .stSelectbox, .stButton, .stForm {
        color: #FFFFFF !important;
    }

    .stSlider label, .stSelectbox label, .stForm label {
        color: #FFFFFF !important;
    }

    .stButton>button {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    .stSelectbox>div>div, .stSelectbox>div>div>ul {
        background-color: #ffffff !important;
        color: #000000 !important;
    }

    .stSelectbox>div>div>ul>li {
        color: #000000 !important;
    }

    .stNumberInput input {
        background-color: #ffffff !important;
        color: #000000 !important;
        border-radius: 6px;
        padding: 5px;
        border: 1px solid #ccc !important;
    }

    /* Extra: number selection highlight */
    .stNumberInput input::selection {
        background: #cce0ff !important;
        color: #000000 !important;
    }

    /* Slider number pop-up fix (the red 30/1000/3 numbers above) */
    .stSlider > div > div > div > span {
        color: #FF4B4B !important;
        font-weight: bold;
    }

    pre, code {
        color: #000000 !important;
        background-color: #ffffff !important;
    }
    </style>
""", unsafe_allow_html=True)


# Load Data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("train.csv")
        mean_arrival_delay = df['Arrival Delay in Minutes'].mean()
        df['Arrival Delay in Minutes'] = df['Arrival Delay in Minutes'].fillna(mean_arrival_delay)
        return df
    except FileNotFoundError:
        st.error("Error: 'train.csv' not found. Please ensure the file is in the correct directory.")
        return None

df = load_data()
if df is None:
    st.stop()

# EDA Section
st.header("Airline Passenger Satisfaction Analysis")

# Data Preview
st.subheader("Data Preview")
st.dataframe(df.head())

# Dataset Info
st.subheader("Dataset Info")
st.write("Shape of dataset:", df.shape)
st.write("Column types:")
st.write(df.dtypes)

# Missing Values
st.subheader("Missing Values")
st.write(df.isnull().sum())

# Summary Stats
st.subheader("Summary Statistics")
st.write(df.describe())

# Visualizations
st.subheader("Key Visualizations")

# Service-related box plots
st.markdown("#### Service Ratings by Satisfaction")
service_cols = ['Inflight wifi service', 'Ease of Online booking', 'Seat comfort', 'Inflight entertainment', 
                'On-board service', 'Leg room service', 'Baggage handling', 'Checkin service', 
                'Inflight service', 'Cleanliness']
fig, axes = plt.subplots(4, 3, figsize=(15, 12))
axes = axes.flatten()
for i, col in enumerate(service_cols):
    sns.boxplot(x='satisfaction', y=col, data=df, ax=axes[i], palette='Set2')
    axes[i].set_title(f'{col} by Satisfaction', color='black')
    axes[i].set_xlabel('Satisfaction', color='black')
    axes[i].set_ylabel(col, color='black')
    axes[i].tick_params(axis='both', colors='black')
# Remove empty subplots
for i in range(len(service_cols), len(axes)):
    fig.delaxes(axes[i])
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Bar plot for Flight Distance by Class
st.markdown("#### Average Flight Distance by Class")
fig2, ax2 = plt.subplots(figsize=(6, 4))
sns.barplot(x='Class', y='Flight Distance', data=df, ax=ax2, palette='Blues_d')
ax2.set_title('Average Flight Distance by Class', color='black')
ax2.set_xlabel('Class', color='black')
ax2.set_ylabel('Flight Distance', color='black')
ax2.tick_params(axis='both', colors='black')
plt.xticks(rotation=45, color='black')
plt.tight_layout()
st.pyplot(fig2)
plt.close()

# Correlation Heatmap
st.markdown("#### Correlation Heatmap")
num_cols = ['Age', 'Flight Distance', 'Departure Delay in Minutes', 'Arrival Delay in Minutes']
corr_matrix = df[num_cols].corr()
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, ax=ax, annot_kws={'color': 'black'})
ax.set_title('Correlation Matrix of Numerical Features', color='black')
ax.tick_params(axis='both', colors='black')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Box Plots for Numerical Features
st.markdown("#### Box Plots of Numerical Features")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes = axes.flatten()
for i, col in enumerate(num_cols):
    sns.boxplot(y=df[col], color='lightgreen', ax=axes[i])
    axes[i].set_title(f'Box Plot of {col}', color='black')
    axes[i].set_ylabel(col, color='black')
    axes[i].tick_params(axis='both', colors='black')
plt.tight_layout()
st.pyplot(fig)
plt.close()

# Histograms for Numerical Features
st.markdown("#### Histograms of Numerical Features")
for col in num_cols:
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.histplot(df[col], bins=30, kde=True, color='skyblue', ax=ax)
    ax.set_title(f'{col} Distribution', color='black')
    ax.set_xlabel(col, color='black')
    ax.set_ylabel('Frequency', color='black')
    ax.tick_params(axis='both', colors='black')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Pie Charts for Categorical Features
st.markdown("#### Pie Charts of Categorical Features")
cat_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class', 'satisfaction']
for col in cat_cols:
    st.markdown(f"**{col} Distribution**")
    st.write(df[col].value_counts())
    fig, ax = plt.subplots(figsize=(8, 6))
    df[col].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=plt.cm.Paired.colors, textprops={'color': 'black'})
    ax.set_title(f'{col} Distribution', color='black')
    ax.set_ylabel('', color='black')
    ax.axis('equal')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Real-time Prediction Section
st.subheader("Real-time Satisfaction Prediction")

# Preprocessing for model
train_df = df.copy()
columns_to_drop = ['Unnamed: 0', 'id']
train_df = train_df.drop(columns=[col for col in columns_to_drop if col in train_df.columns], errors='ignore')
categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']
le_dict = {}

# Fit and transform categorical columns
for col in categorical_cols:
    le = LabelEncoder()
    train_df[col] = le.fit_transform(train_df[col])
    le_dict[col] = le
le_dict['satisfaction'] = LabelEncoder()
train_df['satisfaction'] = le_dict['satisfaction'].fit_transform(train_df['satisfaction'])

numerical_cols = ['Age', 'Flight Distance', 'Inflight wifi service', 'Departure/Arrival time convenient',
                  'Ease of Online booking', 'Gate location', 'Food and drink', 'Online boarding',
                  'Seat comfort', 'Inflight entertainment', 'On-board service', 'Leg room service',
                  'Baggage handling', 'Checkin service', 'Inflight service', 'Cleanliness',
                  'Departure Delay in Minutes', 'Arrival Delay in Minutes']
scaler = StandardScaler()
train_df[numerical_cols] = scaler.fit_transform(train_df[numerical_cols])

# Train-test split
X = train_df.drop('satisfaction', axis=1)
y = train_df['satisfaction']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# User Input for Prediction
st.markdown("<h4 style='color: #FFFFFF;'>🧾 Enter Passenger Details for Prediction</h4>", unsafe_allow_html=True)
col1, col2 = st.columns(2)
with col1:
    age = st.slider("Age", 0, 80, 30)  # Range based on dataset
    flight_distance = st.slider("Flight Distance", 0, 5000, 1000)  # Range based on dataset
    inflight_entertainment = st.slider("Inflight Entertainment Rating", 0, 5, 3)  # Rating scale
with col2:
    gender = st.selectbox("Gender", ['Male', 'Female'])
    customer_type = st.selectbox("Customer Type", ['Loyal Customer', 'disloyal Customer'])
    class_type = st.selectbox("Class", ['Business', 'Eco', 'Eco Plus'])

# Encode user input using the same LabelEncoders
input_data = pd.DataFrame({
    'Gender': [le_dict['Gender'].transform([gender])[0]],
    'Customer Type': [le_dict['Customer Type'].transform([customer_type])[0]],
    'Age': [age],
    'Type of Travel': [le_dict['Type of Travel'].transform(['Business travel'])[0]],  # Default
    'Class': [le_dict['Class'].transform([class_type])[0]],
    'Flight Distance': [flight_distance],
    'Inflight wifi service': [3],  # Default
    'Departure/Arrival time convenient': [3],  # Default
    'Ease of Online booking': [3],  # Default
    'Gate location': [3],  # Default
    'Food and drink': [3],  # Default
    'Online boarding': [3],  # Default
    'Seat comfort': [3],  # Default
    'Inflight entertainment': [inflight_entertainment],
    'On-board service': [3],  # Default
    'Leg room service': [3],  # Default
    'Baggage handling': [3],  # Default
    'Checkin service': [3],  # Default
    'Inflight service': [3],  # Default
    'Cleanliness': [3],  # Default
    'Departure Delay in Minutes': [0],  # Default
    'Arrival Delay in Minutes': [0]  # Default
})

# Ensure input_data columns match X.columns
input_data = input_data[X.columns]

# Scale numerical features
input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])

# Prediction
if st.button("Predict Satisfaction"):
    try:
        prediction = model.predict(input_data)
        satisfaction = le_dict['satisfaction'].inverse_transform(prediction)[0]
        st.markdown(f"<p style='color: #FFFFFF;'>Predicted Satisfaction: <strong>{satisfaction}</strong></p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

# Conclusion
st.subheader("Conclusion")
st.markdown("""
<p style="color: #000000;">

#### Key Findings
- **Service Quality**: Higher ratings in inflight entertainment, seat comfort, and other services correlate with passenger satisfaction.
- **Travel Class**: Business class passengers report longer flight distances and higher satisfaction.
- **Numerical Features**: Flight distance and delays show distinct distributions, with correlations visualized in the heatmap.

#### Practical Implications
- **Actionable Insights**: Airlines can prioritize inflight entertainment and service quality to boost satisfaction.
- **Prediction Tool**: The real-time prediction feature allows users to assess satisfaction based on passenger profiles, aiding decision-making.
</p>
""", unsafe_allow_html=True)
