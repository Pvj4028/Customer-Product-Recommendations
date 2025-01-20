import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import numpy as np
import joblib

# Load product and browsing data
file_path = r'C:\Users\Pavan\Downloads\Reccomendation Project\customer_browsing_history_with_products.csv'
products = pd.read_csv(file_path)

# Encode categorical columns
encoder_category = LabelEncoder()
products['category'] = encoder_category.fit_transform(products['category'])

encoder_brand = LabelEncoder()
products['brand'] = encoder_brand.fit_transform(products['brand'])

encoder_user = LabelEncoder()
products['user_id'] = encoder_user.fit_transform(products['user_id'])

encoder_product = LabelEncoder()
products['product_id'] = encoder_product.fit_transform(products['product_id'])

# Standardize the price column
scaler = StandardScaler()
products['price'] = scaler.fit_transform(products[['price']])

# Define features (X) and target (y)
if 'interaction' not in products.columns:
    np.random.seed(42)
    products['interaction'] = np.random.randint(0, 2, len(products))

X = products[['user_id', 'product_id', 'category', 'brand', 'price']].values
y = products['interaction'].values

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train the SVM model with cross-validation
svm_model = SVC(probability=True, kernel='rbf', C=1, gamma='scale')

# Cross-validation to evaluate performance
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=5)  # 5-fold cross-validation

# Print the cross-validation results
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean():.2f}")
print(f"Standard deviation of cross-validation scores: {cv_scores.std():.2f}")

# Fit the model on the whole training set after cross-validation
svm_model.fit(X_train, y_train)

# Make predictions
y_pred = svm_model.predict(X_test)

# Evaluate the model on the test data
accuracy = accuracy_score(y_test, y_pred)
print(f"SVM Model Accuracy on Test Data: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save the trained SVM model
joblib.dump(svm_model, 'svm_model.pkl')
print("SVM model saved as 'svm_model.pkl'")

# Save encoders and scaler for preprocessing during inference
joblib.dump(encoder_category, 'encoder_category.pkl')
joblib.dump(encoder_brand, 'encoder_brand.pkl')
joblib.dump(encoder_user, 'encoder_user.pkl')
joblib.dump(encoder_product, 'encoder_product.pkl')
joblib.dump(scaler, 'scaler.pkl')
print("Encoders and scaler saved for later use.")


