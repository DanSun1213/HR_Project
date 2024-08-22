
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

    #import socks
    #import socket

def read_hr_data(file_path):
    return pd.read_csv(file_path)



def visualize_education(df: pd.DataFrame) -> plt.Figure:
    plt.figure(figsize=(14, 8))
    
    education_order = ['Below College', 'College', 'Bachelor', 'Master', 'Doctor']
    education_counts = df['Education_Category'].value_counts()
    education_counts = education_counts.reindex(education_order)
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(education_counts)))
    
    ax = sns.barplot(x=education_counts.index, y=education_counts.values, palette=colors, order=education_order)
    ax.set_title('Distribution of Education Levels', fontsize=20, pad=20)
    ax.set_xlabel('Education Level', fontsize=14)
    ax.set_ylabel('Number of Employees', fontsize=14)
    
    for i, v in enumerate(education_counts.values):
        ax.text(i, v + 5, f'{v}\n({v/len(df):.1%})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    total = len(df)
    most_common = education_counts.idxmax()
    least_common = education_counts.idxmin()
    stats_text = f'Total Employees: {total}\nMost Common: {most_common}\nLeast Common: {least_common}'
    plt.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    x = range(len(education_order))
    y = education_counts.values
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", linewidth=2)
    
    plt.xticks(rotation=0)
    plt.tight_layout()
    
    return plt.gcf()

def visualize_job_involvement(df: pd.DataFrame) -> plt.Figure:
    plt.figure(figsize=(14, 8))
    
    involvement_order = ['Low', 'Medium', 'High', 'Very High']
    involvement_counts = df['JobInvolvement_Category'].value_counts()
    involvement_counts = involvement_counts.reindex(involvement_order)
    
    colors = plt.cm.YlOrRd(np.linspace(0.2, 0.8, len(involvement_counts)))
    
    ax = sns.barplot(x=involvement_counts.index, y=involvement_counts.values, palette=colors, order=involvement_order)
    ax.set_title('Distribution of Job Involvement', fontsize=20, pad=20)
    ax.set_xlabel('Involvement Level', fontsize=14)
    ax.set_ylabel('Number of Employees', fontsize=14)
    
    for i, v in enumerate(involvement_counts.values):
        ax.text(i, v + 5, f'{v}\n({v/len(df):.1%})', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    most_common = involvement_counts.idxmax()
    least_common = involvement_counts.idxmin()
    stats_text = f'Total Employees: {len(df)}\nMost Common: {most_common}\nLeast Common: {least_common}'
    plt.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
             verticalalignment='top', horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    x = range(len(involvement_order))
    y = involvement_counts.values
    z = np.polyfit(x, y, 2)
    p = np.poly1d(z)
    plt.plot(x, p(x), "r--", linewidth=2)
    
    plt.tight_layout()
    
    return plt.gcf()


def plot_attrition_rate(df):
    plt.figure(figsize=(10, 8))
    attrition_rate = df['Attrition'].value_counts(normalize=True)
    colors = ['#ff9999', '#66b3ff']
    explode = (0, 0.1)
    
    plt.pie(attrition_rate.values, explode=explode, labels=attrition_rate.index, colors=colors,
            autopct='%1.1f%%', startangle=90, shadow=True, textprops={'fontsize': 14})
    
    plt.title('Overall Attrition Rate', fontsize=18)
    plt.axis('equal')
    plt.legend(title="Attrition", loc="best", bbox_to_anchor=(1, 0, 0.5, 1))
    return plt.gcf()

def plot_correlation_matrix(df):
    numeric_df = df.select_dtypes(include=[np.number])
    correlation_matrix = numeric_df.corr()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', aspect='auto')
    plt.colorbar()
    plt.title('Correlation Matrix of Numeric Features', fontsize=16)
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=90)
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.tight_layout()
    return plt.gcf(), correlation_matrix

def plot_demographic_breakdown(df):
    demographic_cols = ['Age', 'Gender', 'MaritalStatus', 'Education', 'EducationField']
    fig, axes = plt.subplots(3, 2, figsize=(20, 30))
    axes = axes.flatten()
    
    for i, col in enumerate(demographic_cols):
        attrition_by_demo = df.groupby(col)['Attrition'].value_counts(normalize=True).unstack()
        attrition_by_demo.plot(kind='bar', stacked=True, ax=axes[i])
        axes[i].set_title(f'Attrition Rate by {col}', fontsize=14)
        axes[i].set_ylabel('Percentage', fontsize=12)
        axes[i].legend(title='Attrition', loc='upper right')
        axes[i].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    return fig

def plot_salary_analysis(df):
    plt.figure(figsize=(12, 6))
    sns.violinplot(x='Attrition', y='MonthlyIncome', data=df)
    plt.title('Monthly Income Distribution by Attrition', fontsize=16)
    plt.ylabel('Monthly Income', fontsize=12)
    plt.xlabel('Attrition', fontsize=12)
    return plt.gcf()

def plot_job_satisfaction(df):
    plt.figure(figsize=(12, 6))
    sns.countplot(x='JobSatisfaction', hue='Attrition', data=df)
    plt.title('Job Satisfaction vs Attrition', fontsize=16)
    plt.xlabel('Job Satisfaction', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.legend(title='Attrition')
    return plt.gcf()

def plot_years_at_company(df):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Attrition', y='YearsAtCompany', data=df)
    plt.title('Years at Company vs Attrition', fontsize=16)
    plt.xlabel('Attrition', fontsize=12)
    plt.ylabel('Years at Company', fontsize=12)
    return plt.gcf()

def initial_hr_analysis(df):
    plt.style.use('default')
    
    plots = [
        plot_attrition_rate(df),
        plot_correlation_matrix(df),
        plot_demographic_breakdown(df),
        plot_salary_analysis(df),
        plot_job_satisfaction(df),
        plot_years_at_company(df)
    ]
    
    return plots

def apply_filters(df):
    import streamlit as st
    st.sidebar.header("Filters")

    # Department filter
    department = st.sidebar.multiselect("Department", df['Department'].unique())

    # Gender filter
    gender = st.sidebar.multiselect("Gender", df['Gender'].unique())

    # Education filter
    education = st.sidebar.multiselect("Education", df['Education'].unique())

    # Apply filters
    if department:
        df = df[df['Department'].isin(department)]
    
    if gender:
        df = df[df['Gender'].isin(gender)]
    
    if education:
        df = df[df['Education_Category'].isin(education)]

    return df

def plot_employee_attrition(df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17, 6))
    
    # Visualization to show Employee Attrition in Counts
    attrition_rate = df["Attrition"].value_counts()
    sns.barplot(x=attrition_rate.index, y=attrition_rate.values, palette='Set2', ax=ax1)
    #ax1.set_title("Employee Attrition Counts", fontweight="black", size=14, pad=15)
    for i, v in enumerate(attrition_rate.values):
        ax1.text(i, v, v, ha="center", fontsize=14)

    # Visualization to show Employee Attrition in Percentage
    colors = sns.color_palette('Set2', len(attrition_rate))
    ax2.pie(attrition_rate, labels=["No", "Yes"], autopct="%.2f%%", textprops={"size":14},
            colors=colors, explode=[0, 0.1], startangle=90)
    center_circle = plt.Circle((0, 0), 0.3, fc='white')
    ax2.add_artist(center_circle)
    ax2.set_title("Employee Attrition Rate", fontweight="black", size=14, pad=15)

    plt.tight_layout()
    return fig


def pie_bar_plot(df, col, hue):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Extract value counts for the specified column
    value_counts = df[col].value_counts().sort_index()
    
    # First subplot: Pie chart
    ax1.set_title(f"Distribution by {col}", fontweight="black", size=14, pad=15)
    colors = sns.color_palette('Set2', len(value_counts))
    wedges, texts, autotexts = ax1.pie(value_counts.values, labels=value_counts.index, 
                                       autopct="%.1f%%", pctdistance=0.75, startangle=90,
                                       colors=colors, textprops={"size":14})
    center_circle = plt.Circle((0, 0), 0.4, fc='white')
    ax1.add_artist(center_circle)
    
    # Second subplot: Bar plot
    new_df = df[df[hue] == 'Yes']
    value_1 = value_counts
    value_2 = new_df[col].value_counts().sort_index()  # Sort the values in the same order
    attrition_percentages = np.floor((value_2 / value_1) * 100).values
    
    sns.barplot(x=value_2.index, y=value_2.values, palette='Set2', ax=ax2)
    ax2.set_title(f"Attrition Rate by {col}", fontweight="black", size=14, pad=15)
    
    for index, value in enumerate(value_2):
        ax2.text(index, value, f"{value} ({int(attrition_percentages[index])}%)", 
                 ha="center", va="bottom", size=10)
    
    plt.tight_layout()
    return fig

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as ticker

def hist_with_hue(df, col, hue):


    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 6))
    
    # Histogram
    sns.histplot(x=col, hue=hue, data=df, kde=True, palette='Set2', ax=ax1)
    
    # Configure the x-axis to display integer values and center-align the labels
    ax1.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    ax1.set_xticklabels(ax1.get_xticks(), rotation=90, ha='center')
    ax1.set_title(f"Distribution by {col}", fontweight="black", size=14, pad=10)
    
    # Box plot
    sns.boxplot(x=hue, y=col, data=df, palette='Set2', ax=ax2)
    ax2.set_title(f"Distribution by {col} & {hue}", fontweight="black", size=14, pad=10)
    
    plt.tight_layout()
    return fig

def count_percent_plot(df, col, hue):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 8))

    # First subplot: Employees by column
    value_1 = df[col].value_counts()
    sns.barplot(x=value_1.index, y=value_1.values, order=value_1.index, palette='Set2', ax=ax1)
    ax1.set_title(f"Employees by {col}", fontweight="black", size=14, pad=15)
    for index, value in enumerate(value_1.values):
        count_percentage = "{:.1f}%".format((value / len(df)) * 100)
        ax1.text(index, value, f"{value} ({count_percentage})", ha="center", va="bottom", size=10)
    ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)

    # Sort the values for the second subplot to match the order of the first subplot
    value_2 = df[df[hue] == 'Yes'][col].value_counts().reindex(value_1.index)

    # Second subplot: Employee Attrition by column
    attrition_rate = (value_2 / value_1 * 100).values
    sns.barplot(x=value_2.index, y=value_2.values, order=value_1.index, palette='Set2', ax=ax2)
    ax2.set_title(f"Employee Attrition by {col}", fontweight="black", size=14, pad=15)
    for index, value in enumerate(value_2.values):
        attrition_percentage = "{:.1f}%".format(np.round(attrition_rate[index], 1))
        ax2.text(index, value, f"{value} ({attrition_percentage})", ha="center", va="bottom", size=10)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=90)

    plt.tight_layout()
    return fig
def plot_years_at_company_vs_attrition(df):
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(x='Attrition', y='YearsAtCompany', data=df, ax=ax)
    
    ax.set_title('Years at Company vs Attrition', fontsize=16)
    ax.set_xlabel('Attrition', fontsize=12)
    ax.set_ylabel('Years at Company', fontsize=12)
    
    # Add median values on top of each box
    for i, artist in enumerate(ax.artists):
        # The box extends from the first quartile to the third quartile
        # The median is represented by a line inside the box
        median = df[df['Attrition'] == (['No', 'Yes'][i])]['YearsAtCompany'].median()
        ax.text(i, median, f'Median: {median:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    return fig

def local_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

# Function to create the visualization
def create_attrition_plot(df):
    plt.figure(figsize=(10, 20))
    attrition_by_role = df.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack()
    attrition_by_role = attrition_by_role.sort_values('Yes', ascending=False)
    ax = attrition_by_role.plot(kind='bar', stacked=True, width=0.8)
    
    #plt.title('Attrition Rate by Job Role', fontsize=20, pad=20)
    plt.xlabel('Job Role', fontsize=14, labelpad=10)
    plt.ylabel('Percentage', fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend(title='Attrition', fontsize=8, title_fontsize=10, loc='upper left', bbox_to_anchor=(1, 0.1))
    
    for i, role in enumerate(attrition_by_role.index):
        yes_rate = attrition_by_role.loc[role, 'Yes'] * 100
        plt.text(i, 1.02, f'{yes_rate:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    
    summary_text = (f"Total Employees: {len(df)}\n"
                    f"Overall Attrition Rate: {df['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%\n"
                    f"Highest Attrition: {attrition_by_role['Yes'].idxmax()} ({attrition_by_role['Yes'].max()*100:.1f}%)\n"
                    f"Lowest Attrition: {attrition_by_role['Yes'].idxmin()} ({attrition_by_role['Yes'].min()*100:.1f}%)")
    
    plt.text(1.1, 0.98, summary_text, transform=plt.gca().transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return plt

# Function to get AI insights
def get_ai_insights(df):


# Configure SOCKS5 proxy
    #socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
    #socket.socket = socks.socksocket

    data_json = df.to_json(orient='records')
    
    api_key = st.secrets["openai_api_key"]  # Store your API key in Streamlit secrets
    url = 'https://api.openai.com/v1/chat/completions'
    
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    
    data = {
        'model': 'gpt-4',
        'messages': [
            {
                "role": "system",
                "content": """You are an experienced data scientist specializing in HR analytics. 
                You've been asked to analyze employee attrition data for a company. 
                In the data, 'Attrition' column values mean:
                - 'Yes': The employee has left the company
                - 'No': The employee is still active with the company
                Your task is to provide insights on employee attrition rates, identify patterns or factors 
                contributing to attrition, and suggest actionable recommendations to improve employee 
                retention and overall company performance."""
            },
            {
                "role": "user",
                "content": f"""Here's the employee attrition data for analysis:
                {data_json}
                
                Based on this data, please provide:
                1. An overview of the attrition rates across different job roles.
                2. Insights into potential factors contributing to higher attrition in certain roles.
                3. Identification of any patterns or trends in the data related to attrition.
                4. Actionable recommendations to reduce attrition and improve employee retention.
                5. Suggestions on how these changes could positively impact company performance.
                
                Please structure your response with clear headings for each of these points."""
            }
        ],
        'max_tokens': 500,
        'temperature': 0.7
    }
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        result = response.json()
        insights = result['choices'][0]['message']['content']
        return insights
    except Exception as e:
        return f"An error occurred: {e}"


import requests
import base64
import os
import io
import matplotlib.pyplot as plt
import numpy as np

# import socket

# Configure SOCKS5 proxy
#socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
#socket.socket = socks.socksocket

def create_pyplot_image():
    # Create a simple plot
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    ax.plot(x, y)
    ax.set_title("Sine Wave")
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")
    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf

def encode_image(image_buffer):
    return base64.b64encode(image_buffer.getvalue()).decode('utf-8')

def send_image_to_openai(image_buffer, api_key):
    #socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
    #socket.socket = socks.socksocket
    api_key = st.secrets["openai_api_key"]
    base_url = 'https://api.openai.com/v1/chat/completions'
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    base64_image = encode_image(image_buffer)
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this pyplot image. Describe what you see, including the type of plot, axes labels, and any notable features."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 300
    }
    try:
        response = requests.post(base_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}





import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import io
import base64
import requests

def create_attrition_plot1(df):
    plt.figure(figsize=(12, 8))
    sns.set_style("whitegrid")
    sns.set_palette("Set2")

    attrition_by_role = df.groupby('JobRole')['Attrition'].value_counts(normalize=True).unstack()
    attrition_by_role = attrition_by_role.sort_values('Yes', ascending=False)
    
    ax = attrition_by_role.plot(kind='bar', stacked=True, width=0.8)

    #plt.title('Attrition Rate by Job Role', fontsize=20, pad=20)
    plt.xlabel('Job Role', fontsize=14, labelpad=10)
    plt.ylabel('Percentage', fontsize=14, labelpad=10)
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)
    
    # Adjust the plot to make room for labels
    plt.subplots_adjust(right=0.85, top=0.85)

    # Customize legend
    plt.legend(title='Attrition', fontsize=10, title_fontsize=12, 
               loc='center left', bbox_to_anchor=(1.2, 0.5))

    # Add percentage labels above the bars
    for i, role in enumerate(attrition_by_role.index):
        yes_rate = attrition_by_role.loc[role, 'Yes'] * 100
        plt.text(i, 1.01, f'{yes_rate:.1f}%', ha='center', va='bottom', 
                 fontsize=9, fontweight='bold', color='#333333')

    # Set y-axis to go to 105% to make room for labels
    plt.ylim(0, 1.05)

    # Add a thin line at 100%
    plt.axhline(y=1, color='grey', linestyle='--', linewidth=0.8, alpha=0.7)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))

    # Add summary text
    summary_text = (f"Total Employees: {len(df):,}\n"
                    f"Overall Attrition Rate: {df['Attrition'].value_counts(normalize=True)['Yes']*100:.1f}%\n")
                    #f"Highest Attrition: {attrition_by_role['Yes'].idxmax()} ({attrition_by_role['Yes'].max()*100:.1f}%)\n"
                    #f"Lowest Attrition: {attrition_by_role['Yes'].idxmin()} ({attrition_by_role['Yes'].min()*100:.1f}%)")

    plt.text(1.2, 1, summary_text, transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='white', edgecolor='white', alpha=0.8))

    # Save the plot to a buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=300, bbox_inches='tight')
    buf.seek(0)
    plt.close()  # Close the plot to free up memory

    return buf

def encode_image(image_buffer):
    return base64.b64encode(image_buffer.getvalue()).decode('utf-8')

def send_image_to_openai1(image_buffer, api_key):
    base_url = 'https://api.openai.com/v1/chat/completions' # where to send the request
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    } # 
    base64_image = encode_image(image_buffer)
    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": """You are an experienced HR professional analyzing this plot. 
                        Provide insights on the following:
                        \n\n1. Describe the type of plot and what it represents.
                        \n2. Identify the job roles with the highest and lowest attrition rates.
                        \n3. Interpret the overall attrition rate and its implications for the company.
                        \n4. Highlight any concerning trends or patterns in the data.
                        \n5. Based on this data, what are the most pressing issues that need to be addressed?
                        \n6. Provide 2-3 actionable recommendations to reduce attrition in high-risk roles.
                        \n7. Suggest additional data or metrics that would be valuable for a more comprehensive analysis.
                        \n\n  Focus on providing strategic insights that can guide decision-making and improve employee retention.
                        """
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 500
    }
    try:
        response = requests.post(base_url, headers=headers, json=payload)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}
    

# Cache data loading
@st.cache_data
def load_data(file_path):
    return read_hr_data(file_path)

def send_image_to_openai_1(image_file, api_key, prompt):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    image_data = image_file.getvalue()
    base64_image = base64.b64encode(image_data).decode('utf-8')

    payload = {
        "model": "gpt-4-turbo",
        "messages": [
            {
                "role": "system",
                "content": "You are an expert HR data analyst with years of experience in interpreting workforce analytics and providing strategic insights. Your analysis should be thorough, data-driven, and focused on actionable HR strategies."
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 700
    }

    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    return response.json()

def get_data_analysis_prompt():
    return """
    As a seasoned HR data analyst, provide a focused analysis of this visualization for your HR colleague. Your analysis should be practical, strategic, and directly applicable to HR decision-making. Address the following:

    1. Chart Overview:
       • Identify the chart type and its title.
       • Briefly explain how this visualization relates to key HR metrics or challenges.

    2. Data Breakdown:
       • Describe the main components of the chart (axes, categories, etc.).
       • Highlight the top 3 most critical data points, using exact figures or percentages.

    3. HR Implications:
       • What does this data reveal about employee behavior, satisfaction, or performance?
       • Identify any concerning trends or positive developments in workforce dynamics.

    4. Comparative Insights:
       • How do different employee groups or categories compare?
       • Are there any outliers or unexpected patterns that warrant attention?

    5. Strategic HR Focus:
       • What are the 2 most pressing HR issues or opportunities highlighted by this data?
       • How might these insights impact current HR policies or programs?

    6. Action Plan:
       • Suggest 3 specific, data-driven actions HR could take based on this analysis.
       • For each action, briefly explain its potential impact on employee retention, engagement, or productivity.

    7. Future Outlook:
       • Based on the trends shown, what should HR be prepared for in the next 6-12 months?
       • Recommend one additional metric or data point to track that would complement this analysis.

    Present your insights concisely, using bullet points. Focus on information that will help HR make informed decisions about talent management, employee experience, and organizational effectiveness. Your analysis should be both analytical and practical, providing clear guidance for strategic HR initiatives.
    """

def fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    img = Image.open(buf)
    return img

def load_uploaded_data(uploaded_file):
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error reading the file: {str(e)}")
            return None
    return None

def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf.getvalue()

def fig_to_img_buffer(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    buf.seek(0)
    return buf


def sidebar_filters(df):
    st.sidebar.header("Dashboard Controls")

    # Department Filter
    departments = ['All'] + sorted(df['Department'].unique().tolist())
    selected_dept = st.sidebar.selectbox("Filter by Department", departments)

    # Job Role Filter
    job_roles = ['All'] + sorted(df['JobRole'].unique().tolist())
    selected_role = st.sidebar.selectbox("Filter by Job Role", job_roles)

    # Attrition Filter
    attrition_filter = st.sidebar.radio("Show Attrition", ('All', 'Yes', 'No'))

    # Age Range Filter
    age_range = st.sidebar.slider("Age Range", int(df['Age'].min()), int(df['Age'].max()), 
                                  (int(df['Age'].min()), int(df['Age'].max())))

    st.sidebar.markdown("---")
    st.sidebar.caption("Created by Dan Sun, HR Data Analyst")

    # Social Links
    st.sidebar.markdown("### Connect with Me")
    col1, col2 = st.sidebar.columns(2)
    col1.markdown("[![GitHub](https://img.shields.io/badge/GitHub-Profile-blue?style=for-the-badge&logo=github)](https://github.com/DanSun1213)")
    col2.markdown("[![LinkedIn](https://img.shields.io/badge/LinkedIn-Profile-blue?style=for-the-badge&logo=linkedin)](https://www.linkedin.com/in/dan-sun-9b3315186/)")

    return selected_dept, selected_role, attrition_filter, age_range

def apply_filters(df, selected_dept, selected_role, attrition_filter, age_range):
    if selected_dept != 'All':
        df = df[df['Department'] == selected_dept]
    
    if selected_role != 'All':
        df = df[df['JobRole'] == selected_role]
    
    if attrition_filter != 'All':
        df = df[df['Attrition'] == attrition_filter]
    
    df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]
    
    return df


import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import joblib
import pickle
from scipy.special import expit
import io
@st.cache_resource
def load_ml_components():
    try:
        with open('employee_attrition_full_data.pkl', 'rb') as f:
            data_dict = pickle.load(f)
        return data_dict['scaler'], data_dict['pca'], data_dict['best_model'], data_dict['feature_names']
    except:
        st.error("Error loading pickle file. Attempting to load individual components.")
        try:
            scaler = joblib.load('scaler.joblib')
            pca = joblib.load('pca.joblib')
            model = joblib.load('best_model.joblib')
            with open('feature_names.txt', 'r') as f:
                feature_names = [line.strip() for line in f]
            return scaler, pca, model, feature_names
        except FileNotFoundError:
            st.error("Error loading individual components. Please ensure all required files are present.")
            st.stop()

scaler, pca, model, feature_names = load_ml_components()

# Define feature input functions
def get_numeric_input(feature, min_value, max_value, step=1):
    return st.number_input(f'Enter {feature}', min_value=min_value, max_value=max_value, step=step)

def get_categorical_input(feature, options):
    return st.selectbox(f'Select {feature}', options)

def get_binary_input(feature):
    return st.checkbox(feature)

# Define the input form
def employee_input_form():
    st.subheader("Enter Employee Details")
    
    input_data = {}
    
    col1, col2 = st.columns(2)
    
    with col1:
        input_data['Age'] = get_numeric_input('Age', 18, 100)
        input_data['MonthlyIncome'] = get_numeric_input('Monthly Income', 0, 100000, 100)
        input_data['DistanceFromHome'] = get_numeric_input('Distance From Home', 0, 100)
        input_data['TotalWorkingYears'] = get_numeric_input('Total Working Years', 0, 50)
        input_data['YearsAtCompany'] = get_numeric_input('Years at Company', 0, 40)
    
    with col2:
        input_data['EnvironmentSatisfaction'] = get_categorical_input('Environment Satisfaction', [1, 2, 3, 4])
        input_data['JobInvolvement'] = get_categorical_input('Job Involvement', [1, 2, 3, 4])
        input_data['WorkLifeBalance'] = get_categorical_input('Work Life Balance', [1, 2, 3, 4])
        input_data['StockOptionLevel'] = get_categorical_input('Stock Option Level', [0, 1, 2, 3])
        input_data['OverTime'] = get_binary_input('Overtime')
    
    departments = ['Sales', 'Research & Development', 'Human Resources']
    selected_dept = get_categorical_input('Department', departments)
    for dept in departments:
        input_data[f'Department_{dept}'] = 1 if selected_dept == dept else 0
    
    return input_data

def predict_attrition(input_data):
    input_df = pd.DataFrame([input_data])
    
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    input_df = input_df[feature_names]
    scaled_data = scaler.transform(input_df)
    pca_data = pca.transform(scaled_data)
    
    prediction = model.predict(pca_data)[0]
    decision_score = model.decision_function(pca_data)[0]
    probability_score = expit(decision_score)
    attrition_chance = probability_score * 100
    
    attrition_chance = probability_score * 100
    confidence_level = abs(decision_score)  # Use absolute value for confidence
    
    return attrition_chance, confidence_level

def display_prediction_results(attrition_chance, confidence_level):
    st.subheader("Prediction Results")
    
    if attrition_chance > 50:
        st.warning(f'⚠️ Higher risk of attrition. Estimated chance of leaving: {attrition_chance:.1f}%')
    else:
        st.success(f'✅ Lower risk of attrition. Estimated chance of leaving: {attrition_chance:.1f}%')
    
    st.write(f"Model's confidence: {'High' if confidence_level > 1 else 'Moderate' if confidence_level > 0.5 else 'Low'}")
    
    st.info("""
    ### Understanding the Prediction:
    
    Our model uses advanced machine learning techniques to estimate the likelihood of an employee leaving the company. Here's what you need to know:
    
    1. **Prediction**: The model predicts whether an employee is at risk of leaving based on the information provided.
    
    2. **Percentage**: The percentage represents the estimated chance of the employee leaving. A higher percentage indicates a higher risk of attrition.
    
    3. **Confidence**: This indicates how certain the model is about its prediction. Higher confidence means the model is more sure about its estimate.
    
    4. **Model Type**: We use a Support Vector Machine (SVM) model, which is good at classifying employees into "likely to leave" or "likely to stay" categories based on various factors.
    
    5. **Interpretation**: While this prediction can be a useful guide, it should be considered alongside other factors and human judgment in HR decision-making.
    
    Remember, this is a tool to support decision-making, not to replace human insight in managing employees.
    """)