import streamlit as st

st.set_page_config(
    page_title="Financial Assistant",
    page_icon="💰",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'age' not in st.session_state:
    st.session_state.age = 25  # Default age
if 'city_type' not in st.session_state:
    st.session_state.city_type = "Tier_1"  # Default city type
if 'occupation' not in st.session_state:
    st.session_state.occupation = "Professional"  # Default occupation
if 'income' not in st.session_state:
    st.session_state.income = 0
if 'dependents' not in st.session_state:
    st.session_state.dependents = 0
if 'rent' not in st.session_state:
    st.session_state.rent = 0
if 'loan_repayment' not in st.session_state:
    st.session_state.loan_repayment = 0
if 'insurance' not in st.session_state:
    st.session_state.insurance = 0
if 'groceries' not in st.session_state:
    st.session_state.groceries = 0
if 'transport' not in st.session_state:
    st.session_state.transport = 0
if 'eating_out' not in st.session_state:
    st.session_state.eating_out = 0
if 'entertainment' not in st.session_state:
    st.session_state.entertainment = 0
if 'utilities' not in st.session_state:
    st.session_state.utilities = 0
if 'healthcare' not in st.session_state:
    st.session_state.healthcare = 0
if 'education' not in st.session_state:
    st.session_state.education = 0
if 'miscellaneous' not in st.session_state:
    st.session_state.miscellaneous = 0

# Main page content
st.title("Welcome to Your Personal Financial Assistant")
st.write("""
This application helps you analyze your spending habits and provides personalized recommendations 
to improve your financial health. Follow these steps:

1. Take a quick quiz about your basic information
2. Enter your detailed financial information
3. View your personalized financial analysis and recommendations
""")

if st.button("Take Quiz"):
    st.switch_page("pages/quiz.py")
