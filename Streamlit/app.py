# app.py
import streamlit as st
import pandas as pd
import Functions as fn
import plotly.express as px
import plotly.graph_objects as go
import sys
sys.path.append('/data/streamlit')

st.set_page_config(
    page_title="HR Analytics Dashboard",
    page_icon="📊",
    layout="wide",  # This will expand the main content area
    initial_sidebar_state="expanded"
)
def main():

    df = fn.read_hr_data('/Users/skyler/Documents/HR_Project/Data/cleaned_data/full_employee_data.csv')

# Sidebar
st.sidebar.title("Dashboard Navigater")
page = st.sidebar.radio("Navigate", ["Attrition Analysis", "Database"])


df = fn.read_hr_data('/Users/skyler/Documents/HR_Project/Data/cleaned_data/full_employee_data.csv')


filtered_df = fn.apply_filters(df)
    
# Main canvas
if page == "Attrition Analysis":
    st.title("HR Analytics Dashboard")
    
    tab1, tab2, tab3 = st.tabs(["Overview", "Job Characteristics", "Correlation Matrix"])
    
    with tab1:
        df = fn.read_hr_data('/Users/skyler/Documents/HR_Project/Data/cleaned_data/full_employee_data.csv')
        row = st.columns(1)
        with row[0]:
            st.subheader("Employee Attrition Counts and Rate")
            df_attr = df['Attrition']
            job_satisfaction = fn.plot_employee_attrition(df)
            st.pyplot(job_satisfaction)
    # Button for AI analysis
            if st.button("Get AI Insights", key = "ai_insights"):
                with st.spinner("Analyzing data..."):
                    insights = fn.get_ai_insights(df_attr)
                    st.subheader("AI Insights")
                    st.markdown(insights)

        with row[0]:
            st.subheader("Gender Attrition")
            gender = fn.pie_bar_plot(df, 'Gender', 'Attrition')
            st.pyplot(gender)

        with row[0]:
            st.subheader("Education Attrition")
            fig = fn.count_percent_plot(df, 'EducationField', 'Attrition')
            st.pyplot(fig)

        with row[0]:
            st.subheader("Age Attrition")
            age = fn.hist_with_hue(df, 'Age', 'Attrition')
            st.pyplot(age)

        with row[0]:
            st.subheader("Marital Status Attrition")
            gender = fn.pie_bar_plot(df, 'MaritalStatus', 'Attrition')
            st.pyplot(gender)


        #col1, col2 = st.columns(2)
        #with col1:
        with row[0]:
            st.subheader("Years at Company Distribution")
            attriation_rate = fn.plot_years_at_company_vs_attrition(df)
            st.pyplot(attriation_rate)
            
        with row[0]:
            st.subheader("Distance from Home")
            attriation_rate = fn.count_percent_plot(df, 'DistanceGroup', 'Attrition')
            st.pyplot(attriation_rate)

        #with col2:
        with row[0]:
            st.subheader("Salary")
            salary = fn.plot_salary_analysis(df)
            st.pyplot(salary)
        
        with row[0]:
            st.subheader("Education Attrition")
            fig = fn.count_percent_plot(df, 'EducationField', 'Attrition')
            st.pyplot(fig)


    
    with tab2:
        row = st.columns(1)
        #st.header("Job Characteristics")
        with row[0]:
            st.subheader("Job Satisfaction")
            fig = fn.pie_bar_plot(df, 'JobSatisfaction', 'Attrition')
            st.pyplot(fig)

        with row[0]:
            st.subheader("Relationship Satisfaction")
            fig = fn.pie_bar_plot(df, 'RelationshipSatisfaction', 'Attrition')
            st.pyplot(fig)

        with row[0]:
            st.subheader("Work Life balance")
            fig = fn.pie_bar_plot(df, 'WorkLifeBalance', 'Attrition')
            st.pyplot(fig)           

    with tab3:

        df = fn.read_hr_data('/Users/skyler/Documents/HR_Project/Data/cleaned_data/full_employee_data.csv')
        row = st.columns(1)
        with row[0]:
            st.subheader("Correlation Matrix")
            corr = fn.plot_correlation_matrix(df)
            st.pyplot(corr)
        with row[0]:
                # Create and display the plot
            st.subheader("Attrition Rate by Job Role")
            fig = fn.create_attrition_plot1(df)
            st.pyplot(fig)


    # Load the pre-existing CSV file
    job = pd.read_csv('/Users/skyler/Documents/HR_Project/Data/cleaned_data/Job.csv')  # Replace with your actual CSV file name
    
    
    if st.button("Get AI Insights", key="ai_insights_button", help="Click to get AI-powered insights on the attrition data"):
        with st.spinner("Analyzing data..."):

           insights = fn.get_ai_insights(job)
        st.subheader("AI Insights")
        st.markdown(insights)

            

elif page == "Attrition Analysis":
    st.title("Attrition Analysis")
      
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition by Department")
        
    with col2:
        st.subheader("Attrition by Job Role")
        

elif page == "Database":
    st.title("HR Database")
    df = fn.read_hr_data('/Users/skyler/Documents/HR_Project/Data/cleaned_data/full_employee_data.csv')
    st.write(df.head())  # Use st.write() to display the dataframe in Streamlit

    if st.button("Download CSV"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="Download data as CSV",
            data=csv,
            file_name='hr_data.csv',
            mime='text/csv',
        )

if __name__ == '__main__':
    main()


