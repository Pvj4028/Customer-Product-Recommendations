import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained SVM model and preprocessing tools
svm_model = joblib.load('svm_model.pkl')
encoder_category = joblib.load('encoder_category.pkl')
encoder_brand = joblib.load('encoder_brand.pkl')
encoder_user = joblib.load('encoder_user.pkl')
encoder_product = joblib.load('encoder_product.pkl')
scaler = joblib.load('scaler.pkl')

# Load product data
file_path = r'C:\Users\Pavan\Downloads\Reccomendation Project\customer_browsing_history_with_products.csv'
products = pd.read_csv(file_path)

# Define recommendation function using SVM
def recommend_products(user_id, model, top_n=3):
    # Encode user ID
    try:
        encoded_user_id = encoder_user.transform([user_id])[0]
    except ValueError:
        return f"Error: User ID {user_id} not found."

    product_ids = products['product_id'].unique()
    product_scores = []

    # Prepare features and predict interaction probabilities
    for product_id in product_ids:
        # Get product details
        product_info = products[products['product_id'] == product_id].iloc[0]
        try:
            encoded_product_id = encoder_product.transform([product_id])[0]
        except ValueError:
            continue
        category = encoder_category.transform([product_info['category']])[0]
        brand = encoder_brand.transform([product_info['brand']])[0]
        price = scaler.transform([[product_info['price']]])[0][0]

        # Feature vector
        feature_vector = np.array([encoded_user_id, encoded_product_id, category, brand, price]).reshape(1, -1)

        # Predict interaction probability
        score = model.predict_proba(feature_vector)[0][1]
        product_scores.append((product_info['product_name'], product_info['price'], score))

    # Sort products by score and return the top N
    recommended_products = sorted(product_scores, key=lambda x: x[2], reverse=True)[:top_n]
    return recommended_products

# Streamlit UI
def app():
    st.title("Product Recommendation System")

    # Dropdown for selecting user ID from available user IDs in the encoder
    user_id = st.selectbox("Select User ID:", encoder_user.classes_)

    if user_id:
        # Display recommendations for the user
        st.write(f"Recommendations for user: {user_id}")
        recommended_products = recommend_products(user_id, svm_model)

        # Display recommended products
        if isinstance(recommended_products, str):
            st.write(recommended_products)  # Display error message if user ID is invalid
        else:
            for product_name, price, score in recommended_products:
                st.write(f"**{product_name}** - ${price:.2f} ")

# Run the app
if __name__ == '__main__':
    app()




