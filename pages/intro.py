import streamlit as st

def intro():
    st.title("Financial Assistant - Introduction")
    
    st.header("About the Application")
    st.write("""
    Welcome to your personal Financial Assistant! This application helps you:
    - Track your monthly expenses
    - Predict future disposable income
    - Analyze your spending patterns
    - Get personalized recommendations
    """)
    
    st.header("How It Works")
    st.write("""
    1. **Add Your Data**: Enter your financial details including income and expenses
    2. **View Summary**: Get an overview of your financial situation
    3. **Analysis**: See detailed analysis of your spending patterns
    4. **Recommendations**: Receive personalized suggestions for improvement
    """)
    
    st.header("Features")
    st.write("""
    - **Income Prediction**: Uses Linear Regression to predict next month's disposable income
    - **Spending Analysis**: Uses K-Means Clustering to categorize your spending habits
    - **Visual Analytics**: Interactive charts and graphs for better understanding
    - **Personalized Recommendations**: Based on your spending patterns and comparison with similar profiles
    """)
    
    st.header("Getting Started")
    st.write("""
    To begin, click on 'Add Data' to enter your financial information. You'll need to provide:
    - Basic information (age, city type)
    - Income details
    - Monthly expenses across various categories
    """)
    
    if st.button("Back to Home"):
        st.switch_page("app.py")

intro() 