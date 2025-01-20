import pandas as pd

#  Product Data
product_data = {
    'product_id': [1, 2, 3, 4, 5, 6],
    'product_name': ['Laptop', 'Headphones', 'Smartphone', 'Keyboard', 'Tablet', 'Monitor'],
    'category': ['Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics', 'Electronics'],
    'brand': ['BrandA', 'BrandB', 'BrandA', 'BrandC', 'BrandA', 'BrandB'],
    'price': [1000, 150, 600, 100, 400, 300],
}
products = pd.DataFrame(product_data)

# customer browsing data
customer_browsing_data = {
    'user_id': [1, 1, 1, 2, 2, 3, 3, 3, 3],  # Repeated User IDs for more activity
    'product_id': [1, 3, 5, 2, 4, 1, 6, 3, 5]  # Multiple products per user
}
user_browsing_history = pd.DataFrame(customer_browsing_data)

# Print DataFrames
print("Product Data:")
print(products)

print("\nCustomer Browsing Data:")
print(user_browsing_history)

# Merge browsing data with product details
merged_data = user_browsing_history.merge(products, on='product_id', how='left')

# Print merged data to check
print("\nMerged Data:")
print(merged_data)

# Save the merged browsing history with product details to a CSV file
merged_data.to_csv('customer_browsing_history_with_products.csv', index=False)
print("Merged customer browsing data saved to 'customer_browsing_history_with_products.csv'")
