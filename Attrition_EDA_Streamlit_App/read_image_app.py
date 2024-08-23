import streamlit as st
st.set_page_config(page_title="HR Analytics Dashboard", page_icon="📊", layout="wide")
import pandas as pd
from PIL import Image
import Functions as fn
import io
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
import pickle
from scipy.special import expit
import socks  
# Page Configuration


# Constants
DEFAULT_DATA_PATH = './Data/cleaned_data/full_employee_data.csv'



# Data Loading Function
@st.cache_data
def load_data():
    df = pd.read_csv(DEFAULT_DATA_PATH)
    # Add any necessary data preprocessing here
    return df
    
def analyze_image(image_data, api_key, prompt):
    with st.spinner("Analyzing image..."):
        result = fn.send_image_to_openai_1(image_data, api_key, prompt)
    
    if "error" in result:
        st.error(f"An error occurred: {result['error']}")
    else:
        analysis = result['choices'][0]['message']['content']
        st.success("Analysis complete!")
        st.subheader("OpenAI Analysis:")
        st.write(analysis)
        
        st.download_button(
            label="Download Image Analysis",
            data=analysis,
            file_name="image_analysis.txt",
            mime="text/plain"
        )


def sidebar_content(df):
    st.sidebar.header("Navigator")
    
    st.sidebar.markdown("""
    ### Explore Key HR Insights
    
    📊 **Main Features:**
    - Dashboard Overview
    - Demographic Insights
    - Job Satisfaction Trends
    - Salary Distribution
    - Education Impact
    
    🔍 **Special Feature:**  
    - AI-powered image analysis for deeper insights!
    - Machine Learning for predictive employee attrition rate.  

    
    📌 **How to Use:**  
    Navigate through tabs for different analyses.
    
    """)
    
    # Add two filters
    st.sidebar.subheader("Data Filters")
    
    # Department Filter
    departments = ['All'] + sorted(df['Department'].unique().tolist())
    selected_dept = st.sidebar.selectbox("Filter by Department", departments)
    
    # Job Role Filter
    job_roles = ['All'] + sorted(df['JobRole'].unique().tolist())
    selected_role = st.sidebar.selectbox("Filter by Job Role", job_roles)
    
    st.sidebar.caption("Created by Dan Sun, HR Data Analyst")
    
    # Social Links
    st.sidebar.markdown("### Connect with Me")
    col1, col2 = st.sidebar.columns(2)
    col1.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/DanSun1213)")
    col2.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/dan-sun-9b3315186/)")

    return selected_dept, selected_role

def apply_filters(df, selected_dept, selected_role):
    if selected_dept != 'All':
        df = df[df['Department'] == selected_dept]
    
    if selected_role != 'All':
        df = df[df['JobRole'] == selected_role]
    
    return df

# In your main function:


def main():
    st.title("HR Analytics Dashboard")

    # Load data
    df = load_data()

    # Sidebar content and filters
    selected_dept, selected_role = sidebar_content(df)
    
    # Apply filters
    filtered_df = apply_filters(df, selected_dept, selected_role)


    # Main content
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Dashboard", "Exploratory Data Analysis (EDA)", "Predictive Analytics", "Image Analysis", "Data Overview"])

    # Tab 1: Dashboard
    with tab1:
        st.header("HR Analytics Overview")
        
        # Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Employees", len(filtered_df))
        col2.metric("Attrition Rate", f"{filtered_df['Attrition'].value_counts(normalize=True)['Yes']:.2%}")
        col3.metric("Avg Satisfaction", f"{filtered_df['JobSatisfaction'].mean():.2f}")
        col4.metric("Avg Monthly Income", f"${filtered_df['MonthlyIncome'].mean():.2f}")


        # Interactive Attrition by Department Chart

        st.subheader("Attrition by Department")
        fig = px.bar(filtered_df.groupby('Department')['Attrition'].value_counts(normalize=True).unstack(),
                     barmode='group', height=400)
        st.plotly_chart(fig, use_container_width=True)

        # Salary Distribution

        st.subheader("Salary Distribution")
        fig = px.box(filtered_df, x="Department", y="MonthlyIncome", color="Attrition", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Tab 2: EDA (Your existing code with some enhancements)
    with tab2:
        st.header("Exploratory Data Analysis")
        

        # st.subheader("Demographic Insights")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Age Distribution")
            fig_age = fn.hist_with_hue(filtered_df, 'Age', 'Attrition')
            st.pyplot(fig_age)
        
        with col2:
            st.subheader("Education Attrition")
            fig = fn.count_percent_plot(filtered_df, 'EducationField', 'Attrition')
            st.pyplot(fig)

        col3, col4 = st.columns(2)
        with col3:
            st.subheader("Marital Status Attrition")
            gender = fn.pie_bar_plot(filtered_df, 'MaritalStatus', 'Attrition')
            st.pyplot(gender)           

        with col4:
            st.subheader("Job Satisfaction Attrition")
            fig_satisfaction = fn.pie_bar_plot(filtered_df, 'JobSatisfaction', 'Attrition')
            st.pyplot(fig_satisfaction)

        col5, col6 = st.columns(2)
        with col5:
            st.subheader("Job Role Attirtion")
        # Attrition Rate by Job Role
        # st.subheader("Attrition Rate by Job Role")
            image_buffer = fn.create_attrition_plot1(filtered_df)
            st.image(image_buffer, use_column_width=True)

        with col6:
            st.subheader("Monthly Income Attirtion")
            salary = fn.plot_salary_analysis(filtered_df)
            st.pyplot(salary)
        # Your existing code for HR Analysis...

    # Tab 3: Predictive Analytics (New)
    with tab3:
        st.header("Predictive Analytics")
        st.write("This section uses machine learning to predict employee attrition.")
        st.header("Employee Attrition Prediction")
        
        prediction_method = st.radio("Choose input method:", ["Enter Employee Details", "Upload CSV File"])
        
        if prediction_method == "Enter Employee Details":
            input_data = fn.employee_input_form()
            if st.button('Predict Attrition'):
                attrition_chance, confidence_level = fn.predict_attrition(input_data)
                fn.display_prediction_results(attrition_chance, confidence_level)
                

        
        else:
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
            if uploaded_file is not None:
                    input_df = pd.read_csv(uploaded_file)
                    st.write(input_df)
                    
            if st.button('Predict Attrition for Uploaded Data'):
                        results = []
                        for _, row in input_df.iterrows():
                            attrition_chance, _ = fn.predict_attrition(row)
                            results.append(attrition_chance)
                        
                        input_df['Attrition Risk (%)'] = results
                        st.write(input_df)
                        
                        csv = input_df.to_csv(index=False)
                        st.download_button(
                            label="Download results as CSV",
                            data=csv,
                            file_name="attrition_predictions.csv",
                            mime="text/csv",
                        )

        # Add your predictive model here
        # For example, you could have a form where users input employee characteristics
        # and your model predicts the likelihood of attrition

    # Tab 4: Image Analysis (Combination of your Dashboard Image Analysis and Upload Image Analysis)
    with tab4:
        st.header("Image Analysis")
        
        analysis_type = st.radio("Choose analysis type:", ["Analyze Dashboard Images", "Upload Your Own Image"])
        
        if analysis_type == "Analyze Dashboard Images":
            # Your existing dashboard image analysis code
            st.header("Dashboard Image Analysis")
            image_selection = st.selectbox("Select an image to analyze:", 
                                       ["Age Distribution", 
                                        "Education Attrition",
                                        "Marital Status Attrition",
                                        "Job Satisfaction Attrition",
                                        "Job Role Attrition",
                                        "Monthly Income Attrition"])
            if image_selection == "Age Distribution":
                st.pyplot(fig_age)
                image_to_analyze = fn.fig_to_img_buffer(fig_age)
            elif image_selection == "Education Attrition":
                st.pyplot(fig)
                image_to_analyze = fn.fig_to_img_buffer(fig)
            elif image_selection == "Marital Status Attrition":
                st.pyplot(gender)
                image_to_analyze = fn.fig_to_img_buffer(gender)
            elif image_selection == "Job Satisfaction Attrition":
                st.pyplot(fig_satisfaction)
                image_to_analyze = fn.fig_to_img_buffer(fig_satisfaction)
            elif image_selection == "Job Role Attrition":
                st.image(image_buffer, caption="Attrition Rate by Job Role", use_column_width=True)
                image_to_analyze = image_buffer
            else:  # Monthly Income Attrition
                st.pyplot(salary)
                image_to_analyze = fn.fig_to_img_buffer(salary)

            if st.button("Analyze Dashboard Image"):
                api_key = st.secrets["openai_api_key"]
                prompt = fn.get_data_analysis_prompt()
                analyze_image(image_to_analyze, api_key, prompt)

            pass
        else:
            # Your existing upload image analysis code
            st.header("Upload and Analyze Your Own Image")
            uploaded_image = st.file_uploader("Choose an image to analyze...", type=["jpg", "jpeg", "png"])
            if uploaded_image is not None:
                image = Image.open(uploaded_image)
                st.image(image, caption='Uploaded Image', use_column_width=True)
                
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='PNG')
                img_byte_arr.seek(0)
                
                if st.button("Analyze Uploaded Image"):
                    api_key = st.secrets["openai_api_key"]
                    prompt = fn.get_data_analysis_prompt()
                    analyze_image(img_byte_arr, api_key, prompt)
            else:
                st.warning("Please upload an image to analyze.")
                pass

    # Tab 5: Data Overview (Your existing code)
    with tab5:
        # Your existing Data Overview code
        st.header("Data Overview")
        
        # Dataset Information
        st.subheader("About the Dataset")
        st.info("""
        This is a fictional dataset created by IBM data scientists to explore factors that lead to employee attrition. 
        
        With this data, you can explore important questions such as:
        - Show me a breakdown of distance from home by job role and attrition
        - Compare average monthly income by education and attrition
        
        Source: [IBM HR Analytics Employee Attrition & Performance on Kaggle](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset/data)
        """)
        
        st.write("Dataset Preview:")
        st.write(df.head())
        st.write(f"Total records: {len(filtered_df)}")

        # Column Selection
        st.subheader("Select columns to view:")
        all_columns = df.columns.tolist()
        selected_columns = st.multiselect("Choose columns", all_columns, default=all_columns[:5])

        # Display selected columns
        if selected_columns:
            st.write(df[selected_columns].head())
        else:
            st.warning("Please select at least one column to view the data.")

        # Download Options
        st.subheader("Download Data")
        download_option = st.radio("Choose columns to download:", ("All Columns", "Selected Columns Only"))
        
        if download_option == "All Columns":
            csv = df.to_csv(index=False).encode('utf-8')
            file_name = "ibm_hr_analytics_attrition.csv"
        else:
            if selected_columns:
                csv = df[selected_columns].to_csv(index=False).encode('utf-8')
                file_name = "ibm_hr_analytics_attrition_selected.csv"
            else:
                st.warning("Please select at least one column to download the data.")
                st.stop()

        st.download_button(
            label="Download CSV",
            data=csv,
            file_name=file_name,
            mime="text/csv",
        )

        # Dataset Statistics
        st.subheader("Dataset Statistics")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Number of records: {len(df)}")
            st.write(f"Number of columns: {len(df.columns)}")
        with col2:
            st.write(f"Number of numerical columns: {len(df.select_dtypes(include=['int64', 'float64']).columns)}")
            st.write(f"Number of categorical columns: {len(df.select_dtypes(include=['object']).columns)}")
        
        st.write("Column names:", ", ".join(df.columns))
        
        # Basic Statistics
        st.subheader("Basic Statistics for Numerical Columns")
        st.write(df.describe())

        # Additional Dataset Exploration Ideas
        st.subheader("Explore the Dataset")
        st.write("Here are some ideas to explore the dataset:")
        exploration_ideas = [
            "Analyze the relationship between overtime and attrition",
            "Investigate how job satisfaction correlates with years at the company",
            "Examine the impact of work-life balance on attrition rates",
            "Compare attrition rates across different departments",
            "Analyze how performance ratings relate to monthly income"
        ]
        for idea in exploration_ideas:
            st.write(f"- {idea}")


    # Add a footer
        st.markdown("---")
        st.markdown("© 2023 HR Analytics Dashboard. All rights reserved.")

if __name__ == "__main__":
    main()