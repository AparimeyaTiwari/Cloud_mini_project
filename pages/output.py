import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from util.models import load_data, train_kmeans, train_linear_regression, get_spending_analysis

def output():
    st.title("Financial Analysis Dashboard")
    
    # Load and prepare data
    data = load_data()
    
    # Get user data from session state
    user_data = {
        'age': st.session_state.age,
        'city_type': st.session_state.city_type,
        'occupation': st.session_state.occupation,
        'income': st.session_state.income,
        'rent': st.session_state.rent,
        'groceries': st.session_state.groceries,
        'transport': st.session_state.transport,
        'entertainment': st.session_state.entertainment,
        'utilities': st.session_state.utilities,
        'miscellaneous': st.session_state.miscellaneous
    }
    
    # Display user profile
    st.header("User Profile")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Age", user_data['age'])
    with col2:
        st.metric("City Type", user_data['city_type'])
    with col3:
        st.metric("Occupation", user_data['occupation'])
    
    # Train models
    kmeans, scaler, cluster_labels, silhouette_avg = train_kmeans(data)
    expense_models, preprocessor, r2_scores = train_linear_regression(data)
    
    # Display model information
    st.header("Model Information")
    
    # KMeans Clustering Information
    st.subheader("KMeans Clustering")
    st.write(f"Number of clusters: 3")
    st.write(f"Silhouette Score: {silhouette_avg:.3f}")
    st.write("The silhouette score measures how well-separated the clusters are. A score closer to 1 indicates better clustering.")
    
    # Linear Regression Information
    st.subheader("Linear Regression Models")
    st.write("R² scores for each expense category (higher is better):")
    for expense, r2 in r2_scores.items():
        st.write(f"- {expense}: {r2:.3f}")
    st.write("R² score indicates how well the model fits the data. A score of 1 means perfect prediction.")
    
    # Calculate current disposable income
    current_expenses = sum(user_data[expense] for expense in [
        'rent', 'groceries', 'transport', 'entertainment', 'utilities', 'miscellaneous'
    ])
    current_disposable = user_data['income'] - current_expenses
    
    # Prepare user data for prediction
    user_features = pd.DataFrame([{
        'age': user_data['age'],
        'city_type': user_data['city_type'],
        'occupation': user_data['occupation'],
        'income': user_data['income']
    }])
    
    # Transform user features
    X_user = preprocessor.transform(user_features)
    
    # Predict next month's expenses
    predicted_expenses = {}
    for expense, model in expense_models.items():
        predicted_expenses[expense] = model.predict(X_user)[0]
    
    # Calculate predicted disposable income
    predicted_total_expenses = sum(predicted_expenses.values())
    predicted_disposable = user_data['income'] - predicted_total_expenses
    
    # Display financial summary
    st.header("Financial Summary")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Current Disposable Income", f"${current_disposable:,.2f}")
    with col2:
        st.metric("Predicted Next Month Disposable Income", 
                 f"${predicted_disposable:,.2f}",
                 delta=f"${predicted_disposable - current_disposable:,.2f}")
    
    # Display spending distribution
    st.header("Spending Distribution")
    fig, ax = plt.subplots()
    expenses = list(predicted_expenses.keys())
    values = list(predicted_expenses.values())
    ax.pie(values, labels=expenses, autopct='%1.1f%%')
    st.pyplot(fig)
    
    # Display spending comparison
    st.header("Spending Comparison")
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(expenses))
    width = 0.35
    
    current_values = [user_data[expense] for expense in expenses]
    predicted_values = [predicted_expenses[expense] for expense in expenses]
    
    ax.bar(x - width/2, current_values, width, label='Current')
    ax.bar(x + width/2, predicted_values, width, label='Predicted')
    
    ax.set_ylabel('Amount ($)')
    ax.set_title('Current vs Predicted Spending')
    ax.set_xticks(x)
    ax.set_xticklabels(expenses, rotation=45)
    ax.legend()
    
    st.pyplot(fig)
    
    # Get spending analysis
    spending_percentages, overspending, cluster_info = get_spending_analysis(data, user_data, kmeans, scaler)

    
    # Display cluster information
    st.header("Spending Pattern Cluster")
    st.write(f"You belong to Cluster {cluster_info['cluster'] + 1}")
    st.write(f"Number of people in your cluster: {cluster_info['cluster_size']}")
    st.write("Average spending in your cluster:")
    for expense, amount in cluster_info['cluster_stats'].items():
        st.write(f"- {expense}: ${amount:,.2f}")
    
    # Display recommendations
    st.header("Recommendations")
    if overspending:
        st.warning("Areas of potential overspending detected:")
        for category, details in overspending.items():
            st.write(f"- {category}: You're spending ${details['difference']:,.2f} more than average")
            st.write(f"  Current: ${details['user']:,.2f} | Average: ${details['average']:,.2f}")
    else:
        st.success("Your spending patterns are within normal ranges!")
    
    # Add button to start over
    if st.button("Start Over"):
        st.session_state.clear()
        st.experimental_rerun()

output()
