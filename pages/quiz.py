import streamlit as st

def quiz():
    st.title("Financial Habits Quiz")
    
    # Basic Information
    st.header("Basic Information")
    age = st.number_input("Age", min_value=18, max_value=100, value=st.session_state.age)
    city_type = st.selectbox("City Type", ["Tier_1", "Tier_2", "Tier_3"], index=0)
    occupation = st.selectbox("Occupation", ["Retired", "Student", "Self_Employed", "Professional"], index=0)
    
    # Store basic information in session state
    st.session_state.age = age
    st.session_state.city_type = city_type
    st.session_state.occupation = occupation
    
    if st.button("Next: Enter Your Spending Details"):
        st.switch_page("pages/spendings.py")

# Call the quiz function
quiz()
