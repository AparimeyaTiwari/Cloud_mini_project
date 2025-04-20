import streamlit as st

def spendings():
    st.title("Enter Your Financial Details")
    
    # Display basic info from quiz
    st.header("Your Basic Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.write(f"**Age:** {st.session_state.age}")
    with col2:
        st.write(f"**City Type:** {st.session_state.city_type}")
    with col3:
        st.write(f"**Occupation:** {st.session_state.occupation}")
    
    # Financial Information
    st.header("Financial Information")
    income = st.number_input("Monthly Income ($)", min_value=0, value=st.session_state.income)
    dependents = st.number_input("Number of Dependents", min_value=0, value=st.session_state.dependents)
    
    # Fixed Expenses
    st.header("Fixed Monthly Expenses")
    col1, col2, col3 = st.columns(3)
    with col1:
        rent = st.number_input("Rent/Mortgage ($)", min_value=0, value=st.session_state.rent)
    with col2:
        loan_repayment = st.number_input("Loan Repayment ($)", min_value=0, value=st.session_state.loan_repayment)
    with col3:
        insurance = st.number_input("Insurance ($)", min_value=0, value=st.session_state.insurance)
    
    # Variable Expenses
    st.header("Variable Monthly Expenses")
    col1, col2 = st.columns(2)
    with col1:
        groceries = st.number_input("Groceries ($)", min_value=0, value=st.session_state.groceries)
        transport = st.number_input("Transportation ($)", min_value=0, value=st.session_state.transport)
        eating_out = st.number_input("Eating Out ($)", min_value=0, value=st.session_state.eating_out)
        entertainment = st.number_input("Entertainment ($)", min_value=0, value=st.session_state.entertainment)
        utilities = st.number_input("Utilities ($)", min_value=0, value=st.session_state.utilities)
    with col2:
        healthcare = st.number_input("Healthcare ($)", min_value=0, value=st.session_state.healthcare)
        education = st.number_input("Education ($)", min_value=0, value=st.session_state.education)
        miscellaneous = st.number_input("Miscellaneous ($)", min_value=0, value=st.session_state.miscellaneous)
    
    # Store financial information in session state
    st.session_state.income = income
    st.session_state.dependents = dependents
    st.session_state.rent = rent
    st.session_state.loan_repayment = loan_repayment
    st.session_state.insurance = insurance
    st.session_state.groceries = groceries
    st.session_state.transport = transport
    st.session_state.eating_out = eating_out
    st.session_state.entertainment = entertainment
    st.session_state.utilities = utilities
    st.session_state.healthcare = healthcare
    st.session_state.education = education
    st.session_state.miscellaneous = miscellaneous
    
    # Navigation buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Back to Quiz"):
            st.switch_page("pages/quiz.py")
    with col2:
        if st.button("View Analysis"):
            st.switch_page("pages/output.py")

# Call the spendings function
spendings()
