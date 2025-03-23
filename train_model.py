import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load the dataset
df = pd.read_csv("autism_screening.csv")

# Handle missing values
#drop the outliers
df = df.drop(df[df['age'] == 383].index)
df['age'] = df['age'].fillna(round(df['age'].mean()))

df['ethnicity'] = df['ethnicity'].replace('?', 'others')
df['relation'] = df['relation'].replace('?', 'Others')

#drop the 'age_desc' column
df = df.drop('age_desc', axis=1)


# Encode categorical variables
label_encoders = {}
for col in ['gender', 'ethnicity', 'jundice', 'austim', 'contry_of_res', 'used_app_before', 'relation']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoders for future use
print(df.dtypes)
print(df.select_dtypes(include=['object']).head())

# Define features and target
X = df.drop(columns=["Class/ASD"])
y = LabelEncoder().fit_transform(df["Class/ASD"])

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a deep learning model
model = Sequential([
    Dense(10, activation='relu', input_dim=X.shape[1]),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=150, validation_split=0.25)

# Save the model
model.save("asd_model.h5")

# Save label encoders
joblib.dump(label_encoders, "label_encoders.pkl")
