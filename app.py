import streamlit as st
# from altair.vegalite.v4.api import Chart



def main():
    st.title("Loan Prediction Based on Customer Behaviour")
    st.sidebar.title("Loan")

# Run the app
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass

