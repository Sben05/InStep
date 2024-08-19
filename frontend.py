# Main Author: Shreeniket Bendre
# Northwestern University
# Infosys InStep
# Jun 24 2024
# frontend.py

import streamlit as st
from PIL import Image
from backend import upload_image_to_imgbb, format_response, analyze_image
from pytrends.request import TrendReq
from datetime import datetime
import urllib.request
import requests
from time import sleep
from stqdm import stqdm
from streamlit_custom_notification_box import custom_notification_box
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore
import json
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from streamlit_option_menu import option_menu
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import time
import altair as alt
from datetime import datetime, timedelta
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import sendgrid
from sendgrid.helpers.mail import Mail, Email, To, Content
import streamlit as st
import os
import streamlit as st
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email_validator import validate_email, EmailNotValidError
from email import encoders
from io import BytesIO


st.set_page_config(page_title="Inventory Management Dashboard", layout="wide")

# Function to upload and display image
def upload_and_display_image():
    st.header("Capture a product image to upload")
    img = st.camera_input("Please allow camera permissions if necessary")
    #if img is not None:
     #   st.image(img, caption="Captured Image", use_column_width=True)
      #  for _ in stqdm(range(50), desc="Processing Image", mininterval=1):
       #   sleep(1)
    return img


##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################


# Function to get the current date
def get_current_date():
    st.header("Enter date of product initial shelf stockage")
    curDate = st.date_input(
        "When ready press proceed to access camera upload",
        value=None,
        min_value=None,
        max_value=None,
        key=None,
        help=None,
        on_change=None,
        args=None,
        kwargs=None,
        format="YYYY-MM-DD",
        disabled=False,
        label_visibility="visible"
    )
    date_time = None
    try:
      date_time = curDate.strftime("%Y/%m/%d")
    except:
      pass
    return date_time

# Function to get image value
def get_image_value(img):
    if img is not None:
        return img.getvalue()
    return None

# Function to get and upload logo
def get_logo(url):

    # This statement requests the resource at
    # the given link, extracts its contents
    # and saves it in a variable
    data = requests.get(url).content

    # Opening a new file named img with extension .jpg
    # This file would store the data of the image file
    f = open('logo.png','wb')

    # Storing the image data inside the data variable to the file
    f.write(data)
    f.close()

    if open:
      st.logo("logo.png")

# Function to upload image to imgbb and get the URL
def upload_image(val, imgbb_api_key):
    if val is not None:
        image_url = upload_image_to_imgbb(val, imgbb_api_key)
        if image_url:
            for _ in stqdm(range(10), desc="Processing Image", mininterval=1):
                sleep(0.3)
            #st.link_button("Image URL", image_url)
            styles = {
                      'material-icons': {'color': 'white'},
                      'text-icon-link-close-container': {'box-shadow': '#3896de 0px 4px'},
                      'notification-text': {'font-family': 'monaco'},
                      'close-button': {'font-family': 'monaco'},
                      'link': {'font-family': 'monaco'}
                      }

            custom_notification_box(icon='open_in_new', textDisplay='Successfully Uploaded!', externalLink='Link', url=image_url, styles=styles, key="foo")
            return image_url
        else:
            st.error("Failed to upload image. Please try again.")
    return None

# Function to analyze the image
def analyze_uploaded_image(image_url, date_time):
    if image_url:
        with st.spinner('Analyzing the image...'):
            response_content = analyze_image(image_url, date_time)
        st.success('Analysis complete!')
        return response_content
    return None

# Function to format and display the response
def display_analysis_results(response_content):
    if response_content:
        formatted_response = format_response(response_content)
        st.markdown(formatted_response)
    else:
        st.error("Failed to parse the response. Please ensure the model is providing a valid JSON response.")
    return None

# Function to fetch all categories from Firebase
def fetch_all_categories(db):
    collections = db.collections()
    return [collection.id for collection in collections]


##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################


# Function to fetch data from Firebase
def fetch_data_from_firebase(db, category):
    collection_ref = db.collection(category)
    docs = collection_ref.stream()
    data = [doc.to_dict() for doc in docs]
    return data

# Function to display all products in a grid format
def display_products_grid(db):
    categories = fetch_all_categories(db)
    for category in categories:
        st.subheader(f"Category: {category}")
        products = fetch_data_from_firebase(db, category)
        cols = st.columns(4)  # Adjust the number of columns based on the desired grid layout
        for index, product in enumerate(products):
            with cols[index % 4]:
                st.image(product["Image URL"], use_column_width=True)
                st.write(product["Product name"])
                if st.button("View Details", key=f"{product['id']}"):
                    st.session_state["selected_product"] = product
                    st.session_state["page"] = "Product Details"

# Function to display detailed information about a product
def display_product_details():
    product = st.session_state.get("selected_product", {})
    if product:
        st.image(product["Image URL"], use_column_width=True)
        st.write("### Product Details")
        for key, value in product.items():
            if key != "Image URL" and key != "id":
                st.write(f"**{key.replace('_', ' ').capitalize()}**: {value}")
        if st.button("Back to Product List"):
              st.session_state["page"] = "Review Data"


##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################


def display_home_page():


    # Sample data with real-world perishable products
    data = {
        "Product Name": [
            "Groceries", "Household Items", "Personal Care",
            "Clothing and Accessories", "Electronics", "Home and Garden",
            "Toys and Games", "Automotive", "Sporting Goods",
            "Books and Stationery", "Pharmacy and Health", "Pet Supplies"
        ],
        "Stock": [50, 120, 90, 200, 40, 70, 60, 80, 110, 150, 95, 130],
        "Sales": [45, 100, 80, 180, 30, 65, 50, 70, 100, 140, 90, 120],
        "Shelf Life (days)": [180, 365, 730, 730, 1095, 1825, 1095, 365, 365, 1825, 730, 365],
        "Category": [
            "Groceries", "Household Items", "Personal Care",
            "Clothing and Accessories", "Electronics", "Home and Garden",
            "Toys and Games", "Automotive", "Sporting Goods",
            "Books and Stationery", "Pharmacy and Health", "Pet Supplies"
        ],
        "Initial Stock Date": pd.to_datetime([
            "2023-07-01", "2023-07-02", "2023-07-03", "2023-07-04",
            "2023-07-05", "2023-07-06", "2023-07-07", "2023-07-08",
            "2023-07-09", "2023-07-10", "2023-07-11", "2023-07-12"
        ])
    }


    df = pd.DataFrame(data)

    # Welcome Section
    st.markdown(
        """
        <style>
        .welcome-section {
            text-align: center;
            margin-bottom: 20px;
        }
        .welcome-title {
            color: #FF6F9D;
            font-size: 36px;
            font-weight: bold;
        }
        .welcome-subtitle {
            color: #4a4a4a;
            font-size: 24px;
        }
        </style>
        <div class="welcome-section">
            <div class="welcome-title">Hello!</div>
            <div class="welcome-subtitle">Welcome to the Inventory Management Dashboard</div>
        </div>
        """, unsafe_allow_html=True
    )

    # Big Eye-Catching Number
    st.markdown(
        """
        <style>
        .metric-section {
            background-color: #ffecb3;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            text-align: center;
        }
        .metric-title {
            color: #ff9800;
            font-size: 30px;
            font-weight: bold;
        }
        .metric-value {
            color: #4CAF50;
            font-size: 50px;
            font-weight: bold;
        }
        </style>
        <div class="metric-section">
            <div class="metric-title">Estimated Savings on Perishables:</div>
            <div class="metric-value">15% Saved</div>
        </div>
        """, unsafe_allow_html=True
    )

    # Create columns for dynamic side-by-side layout for three charts
    col1, col2, col3 = st.columns(3)

    with col1:
        # Dot plot for Stock vs Sales
        st.write("### Stock vs Sales")
        stock_sales_chart = alt.Chart(df).mark_point(filled=True, size=100).encode(
            x=alt.X('Stock:Q', title='Stock'),
            y=alt.Y('Sales:Q', title='Sales'),
            color=alt.Color('Category:N', legend=None),
            tooltip=['Product Name:N', 'Stock:Q', 'Sales:Q', 'Shelf Life (days):Q']
        ).properties(
            width=300,
            height=300
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16,
            anchor='start'
        )
        st.altair_chart(stock_sales_chart)

        st.markdown(
            """
            <div style="background-color: #fff3e0; border: 1px solid #ffab91; border-radius: 8px; padding: 15px;">
                <div style="font-size: 18px; color: #ff6f61; font-weight: bold;">
                    Stock vs Sales
                </div>
                <div style="font-size: 14px; color: #4a4a4a;">
                    Visualizes the relationship between stock levels and sales figures for each product.
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    with col2:
        # Bar chart for Stock by Product
        st.write("### Stock by Product")
        stock_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Product Name:N', title='Product Name'),
            y=alt.Y('Stock:Q', title='Stock'),
            color=alt.Color('Product Name:N', legend=None)
        ).properties(
            width=300,
            height=300
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16,
            anchor='start'
        )
        st.altair_chart(stock_chart)

        st.markdown(
            """
            <div style="background-color: #fff3e0; border: 1px solid #ffab91; border-radius: 8px; padding: 15px;">
                <div style="font-size: 18px; color: #ff6f61; font-weight: bold;">
                    Stock by Product
                </div>
                <div style="font-size: 14px; color: #4a4a4a;">
                    Displays the stock levels of each product.
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    with col3:
        # Bar chart for Sales by Product
        st.write("### Sales by Product")
        sales_chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('Product Name:N', title='Product Name'),
            y=alt.Y('Sales:Q', title='Sales'),
            color=alt.Color('Product Name:N', legend=None)
        ).properties(
            width=300,
            height=300
        ).configure_axis(
            labelFontSize=12,
            titleFontSize=14
        ).configure_title(
            fontSize=16,
            anchor='start'
        )
        st.altair_chart(sales_chart)

        st.markdown(
            """
            <div style="background-color: #fce4ec; border: 1px solid #f48fb1; border-radius: 8px; padding: 15px;">
                <div style="font-size: 18px; color: #d81b60; font-weight: bold;">
                    Sales by Product
                </div>
                <div style="font-size: 14px; color: #4a4a4a;">
                    Illustrates the sales figures for each product.
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    # Full-width Line chart for Stock Over Time
    st.write("### Stock Over Time")
    stock_time_chart = alt.Chart(df).mark_line().encode(
        x=alt.X('Initial Stock Date:T', title='Date'),
        y=alt.Y('Stock:Q', title='Stock'),
        color=alt.value('green')
    ).properties(
        width=1200,
        height=300
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='start'
    )
    st.altair_chart(stock_time_chart)

    st.markdown(
        """
        <div style="background-color: #f3e5f5; border: 1px solid #ce93d8; border-radius: 8px; padding: 15px;">
            <div style="font-size: 18px; color: #ab47bc; font-weight: bold;">
                Stock Over Time
            </div>
            <div style="font-size: 14px; color: #4a4a4a;">
                Tracks stock levels over time to monitor changes.
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    # Circular Graphs
    st.write("### Inventory Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        # Circle chart for Stock
        stock_circular_chart = alt.Chart(df).mark_arc(innerRadius=50, outerRadius=100).encode(
            theta=alt.Theta(field="Stock", type="quantitative"),
            color=alt.Color(field="Product Name", type="nominal"),
            tooltip=["Product Name:N", "Stock:Q"]
        ).properties(
            width=300,
            height=300
        ).configure_legend(
            labelFontSize=12,
            titleFontSize=14
        )
        st.altair_chart(stock_circular_chart)

        st.markdown(
            """
            <div style="background-color: #fff3e0; border: 1px solid #ffab91; border-radius: 8px; padding: 15px;">
                <div style="font-size: 18px; color: #ff6f61; font-weight: bold;">
                    Stock Distribution
                </div>
                <div style="font-size: 14px; color: #4a4a4a;">
                    Shows the distribution of stock across different products.
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    with col2:
        # Circle chart for Sales
        sales_circular_chart = alt.Chart(df).mark_arc(innerRadius=50, outerRadius=100).encode(
            theta=alt.Theta(field="Sales", type="quantitative"),
            color=alt.Color(field="Product Name", type="nominal"),
            tooltip=["Product Name:N", "Sales:Q"]
        ).properties(
            width=300,
            height=300
        ).configure_legend(
            labelFontSize=12,
            titleFontSize=14
        )
        st.altair_chart(sales_circular_chart)

        st.markdown(
            """
            <div style="background-color: #e1f5fe; border: 1px solid #81d4fa; border-radius: 8px; padding: 15px;">
                <div style="font-size: 18px; color: #039be5; font-weight: bold;">
                    Sales Distribution
                </div>
                <div style="font-size: 14px; color: #4a4a4a;">
                    Displays the distribution of sales across different products.
                </div>
            </div>
            """, unsafe_allow_html=True
        )

    # Placeholder for Sales Forecasting Chart
    st.write("### Sales Forecasting")
    st.markdown(
        """
        <div style="background-color: #e8f5e9; border: 1px solid #a5d6a7; border-radius: 8px; padding: 15px;">
            <div style="font-size: 18px; color: #388e3c; font-weight: bold;">
                Sales Forecasting
            </div>
            <div style="font-size: 14px; color: #4a4a4a;">
                Placeholder for sales forecasting graph.
            </div>
        </div>
        """, unsafe_allow_html=True
    )
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################


def train_and_save_model():
    # Automated generation of training data
    np.random.seed(42)
    train_data = {
        "expiry_date": [],
        "sales_velocity": [],
        "stock_level": [],
        "percent_discounted": [],
        "days_until_expiry": [],
        "expiry_risk": []
    }

    for _ in range(100):
        train_data["expiry_date"].append(pd.Timestamp('2024-07-01') + pd.to_timedelta(np.random.randint(0, 100), unit='d'))
        train_data["sales_velocity"].append(1.5 + np.random.normal(0, 1.5))
        train_data["stock_level"].append(100 + np.random.randint(-50, 100))
        train_data["percent_discounted"].append(np.random.uniform(0, 50))
        train_data["days_until_expiry"].append(np.random.randint(1, 100))
        train_data["expiry_risk"].append(np.random.randint(0, 2))

    # Convert training data to DataFrame
    df_train = pd.DataFrame(train_data)
    df_train['expiry_date'] = pd.to_datetime(df_train['expiry_date'])

    # Extract features and target variable for training
    X_train = df_train[['sales_velocity', 'stock_level', 'days_until_expiry', 'percent_discounted']]
    y_train = df_train['expiry_risk']

    # Initialize and train the logistic regression model
    expiry_model = LogisticRegression()
    expiry_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(expiry_model, 'expiry_model.joblib')


##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################

# Function to display expiry risk analysis page
def display_expiry_risk_analysis():
    st.title("Expiry Risk Analysis")

    # Pretend loading calculations
    st.write("Initializing model training...")
    progress_bar = st.progress(0)
    for i in range(10):
        time.sleep(0.1)
        progress_bar.progress((i + 1) * 10)

    # Train and save the model
    train_and_save_model()
    st.success("Model training complete!")

    # Pretend more calculations
    st.write("Generating product data...")
    progress_bar = st.progress(0)
    for i in range(10):
        time.sleep(0.1)
        progress_bar.progress((i + 1) * 10)

    # List of real grocery product names
    grocery_names = [
        "Milk", "Bread", "Eggs", "Butter", "Cheese",
        "Yogurt", "Chicken", "Beef", "Pork", "Fish",
        "Apples", "Bananas", "Oranges", "Tomatoes", "Potatoes"
    ]
    grocery_names = [
            "Groceries", "Household Items", "Personal Care",
            "Clothing and Accessories", "Electronics", "Home and Garden",
            "Toys and Games", "Automotive", "Sporting Goods",
            "Books and Stationery", "Pharmacy and Health", "Pet Supplies",
            "Oranges", "Tomatoes", "Potatoes"
        ]
    # Generate similar data for 15 different grocery items
    np.random.seed(42)
    grocery_items = [
        {
            "expiry_date": pd.Timestamp('2024-07-01') + pd.to_timedelta(np.random.randint(0, 100), unit='d'),
            "sales_velocity": 1.935246582 + np.random.normal(0, 0.5),
            "stock_level": 59 + np.random.randint(-10, 10),
            "days_until_expiry": 14 + np.random.randint(-5, 5),
            "Product_name": grocery_names[i]
        }
        for i in range(15)
    ]

    # Convert to DataFrame
    df_pred = pd.DataFrame(grocery_items)

    st.success("Product data generation complete!")

    # Pretend loading model
    st.write("Loading trained model...")
    progress_bar = st.progress(0)
    for i in range(10):
        time.sleep(0.1)
        progress_bar.progress((i + 1) * 10)

    # Load trained model
    expiry_model = joblib.load('expiry_model.joblib')
    st.success("Model loaded!")

    # Normalize the stock level in prediction data
    df_pred['norm_stock_level'] = df_pred['stock_level'] / df_pred['stock_level'].max()

    # Select features for prediction
    X_pred = df_pred[['sales_velocity', 'stock_level', 'days_until_expiry']]

    # Add a placeholder column for percent_discounted to match the training feature set
    X_pred['percent_discounted'] = 0

    # Pretend making predictions
    st.write("Calculating expiry risks...")
    progress_bar = st.progress(0)
    for i in range(10):
        time.sleep(0.1)
        progress_bar.progress((i + 1) * 10)

    # Predict expiry risk probabilities for the new data
    y_pred_prob = expiry_model.predict_proba(X_pred)[:, 1]

    # Add the predicted probabilities to the new data
    df_pred['predicted_expiry_risk'] = y_pred_prob

    # Save the prediction results
    df_pred.to_csv('predicted_risk.csv', index=False)
    st.write(f"Predicted risk data saved to predicted_risk.csv")

    # Select a product for analysis
    selected_product = st.selectbox("Select a product", df_pred['Product_name'].unique())

    # Filter the DataFrame to only include the selected product
    df_selected = df_pred[df_pred['Product_name'] == selected_product].copy()

    # Display the risk calculation
    st.write(f"Predicted Expiry Risk for {selected_product}: {df_selected['predicted_expiry_risk'].values[0]:.2f}")

    # Price elasticity of demand (always negative for this example)
    price_elasticity_of_demand = -1.5  # Example value
    price_per_item = 10  # Example value

    # Calculate revenue without discount
    df_selected['initial_units_sold'] = df_selected['sales_velocity'] * df_selected['days_until_expiry']
    df_selected['revenue_no_discount'] = df_selected['initial_units_sold'] * price_per_item

    # Constants for the discount calculation
    alpha = 0.05  # Scaling constant for days until expiry
    beta = 0.01   # Scaling constant for stock level
    gamma = 0.1   # Scaling constant for sales velocity

    # Calculate the recommended discount dynamically
    df_selected['recommended_discount'] = (
        alpha * df_selected['days_until_expiry'] +
        beta * df_selected['stock_level'] +
        gamma * df_selected['sales_velocity']
    )

    # Adjust the recommended discount based on price and price elasticity of demand
    df_selected['recommended_discount'] *= price_elasticity_of_demand * price_per_item

    # Ensure the discount does not exceed a reasonable percentage (e.g., 50%)
    df_selected['recommended_discount'] = df_selected['recommended_discount'].clip(upper=50)

    # Calculate the percentage change in price due to discount
    df_selected['percentage_change_in_price'] = -df_selected['recommended_discount']

    # Calculate the percentage change in quantity demanded
    df_selected['percentage_change_in_quantity_demanded'] = price_elasticity_of_demand * (df_selected['percentage_change_in_price'] / 100)

    # Estimate the increase in units sold
    df_selected['increase_in_units_sold'] = df_selected['initial_units_sold'] * (df_selected['percentage_change_in_quantity_demanded'] / 100)

    # Calculate the new total units sold
    df_selected['new_units_sold'] = df_selected['initial_units_sold'] + df_selected['increase_in_units_sold']

    # Calculate the discounted price per item
    df_selected['discounted_price'] = price_per_item * (1 - df_selected['recommended_discount'] / 100)

    # Calculate the expected revenue with discount
    df_selected['expected_revenue_with_discount'] = df_selected['new_units_sold'] * df_selected['discounted_price']

    # Create a DataFrame for the chart
    chart_data = pd.DataFrame({
        'Metric': ['Revenue Without Discount', 'Revenue With Discount'],
        'Revenue': [df_selected['revenue_no_discount'].values[0], df_selected['expected_revenue_with_discount'].values[0]]
    })

    # Bar chart for revenue comparison
    discounts_with_percent = ', '.join(abs(df_selected['recommended_discount']).round(0).astype(str) + '%')


    revenue_chart = alt.Chart(chart_data).mark_bar().encode(
        x=alt.X('Metric', sort=None, axis=alt.Axis(labelAngle=0)),
        y='Revenue',
        color=alt.Color('Metric', scale=alt.Scale(range=['#FF6F61', '#6B5B95']))
    ).properties(
        title = f'Expected Revenue Comparison for {selected_product} \nRecommended Discount: ' + discounts_with_percent
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='start'
    )
    savings = df_selected['expected_revenue_with_discount'].values[0] - df_selected['revenue_no_discount'].values[0]

    # Display the chart
    st.altair_chart(revenue_chart, use_container_width=True)

    st.write(f"Estimated Savings: ${savings:.2f}")

    # Scatter plot for expiry risk vs stock level
    st.write("### Expiry Risk vs Stock Level")
    expiry_risk_chart = alt.Chart(df_pred).mark_point(filled=True, size=100).encode(
        x=alt.X('stock_level:Q', title='Stock Level'),
        y=alt.Y('predicted_expiry_risk:Q', title='Predicted Expiry Risk'),
        color=alt.Color('Product_name:N', legend=None),
        tooltip=['Product_name:N', 'stock_level:Q', 'predicted_expiry_risk:Q']
    ).properties(
        width=600,
        height=300
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='start'
    )
    st.altair_chart(expiry_risk_chart)

    st.markdown(
        """
        <div style="background-color: #f3e5f5; border: 1px solid #ce93d8; border-radius: 8px; padding: 15px;">
            <div style="font-size: 18px; color: #ab47bc; font-weight: bold;">
                Expiry Risk vs Stock Level
            </div>
            <div style="font-size: 14px; color: #4a4a4a;">
                This scatter plot visualizes the <span style="color: #ab47bc;">predicted expiry risk</span>
                in relation to the <span style="color: #ab47bc;">stock level</span> for each product.
                It helps to identify products that may have a higher risk of expiry based on their stock levels.
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    # Histogram for days until expiry
    st.write("### Distribution of Days Until Expiry")
    days_until_expiry_chart = alt.Chart(df_pred).mark_bar().encode(
        x=alt.X('days_until_expiry:Q', bin=alt.Bin(maxbins=10), title='Days Until Expiry'),
        y=alt.Y('count():Q', title='Frequency'),
        color=alt.value('lightcoral')
    ).properties(
        width=600,
        height=300
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='start'
    )
    st.altair_chart(days_until_expiry_chart)

    st.markdown(
        """
        <div style="background-color: #fce4ec; border: 1px solid #f48fb1; border-radius: 8px; padding: 15px;">
            <div style="font-size: 18px; color: #d81b60; font-weight: bold;">
                Distribution of Days Until Expiry
            </div>
            <div style="font-size: 14px; color: #4a4a4a;">
                This histogram displays the <span style="color: #d81b60;">distribution of days until expiry</span>
                for the products. It provides insights into how much time is remaining before products expire.
            </div>
        </div>
        """, unsafe_allow_html=True
    )

    # Box plot for sales velocity distribution
    st.write("### Sales Velocity Distribution")
    sales_velocity_chart = alt.Chart(df_pred).mark_boxplot().encode(
        x=alt.X('Product_name:N', title='Product Name'),
        y=alt.Y('sales_velocity:Q', title='Sales Velocity'),
        color=alt.value('mediumseagreen')
    ).properties(
        width=600,
        height=300
    ).configure_axis(
        labelFontSize=12,
        titleFontSize=14
    ).configure_title(
        fontSize=16,
        anchor='start'
    )
    st.altair_chart(sales_velocity_chart)

    st.markdown(
        """
        <div style="background-color: #e8f5e9; border: 1px solid #c8e6c9; border-radius: 8px; padding: 15px;">
            <div style="font-size: 18px; color: #43a047; font-weight: bold;">
                Sales Velocity Distribution
            </div>
            <div style="font-size: 14px; color: #4a4a4a;">
                This box plot illustrates the <span style="color: #43a047;">distribution of sales velocity</span>
                across different products. It helps in understanding the variability in sales speed.
            </div>
        </div>
        """, unsafe_allow_html=True
    )

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
def display_notification_manager():
    # Main page content with new styling
    st.markdown("""
    <style>
        body {
            background-color: #1A1A1D; /* Dark background color */
            color: #FFD1DC; /* Light pink text color */
            font-family: 'Courier New', monospace; /* Monospace font */
        }
        .header {
            font-size: 36px;
            font-weight: bold;
            color: #FF6F9D; /* Light pinkish color */
            text-align: center;
            margin: 30px auto;
            width: 80%;
            background-color: #2E2E3A; /* Darker background for header */
            padding: 15px;
            border-radius: 15px;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
            animation: fadeIn 1.5s ease-in-out;
        }
        .description-box {
            background-color: #2E2E3A; /* Darker secondary background color */
            border-radius: 20px;
            padding: 20px;
            margin-bottom: 30px;
            font-size: 16px;
            color: #FFF1FA; /* Light pink text color */
            font-weight: bold;
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
            animation: slideIn 1s ease-in-out;
        }
        .description-box span {
            color: #FF6F9D; /* Light pinkish color for key points */
        }
        .input-container {
            display: flex;
            flex-direction: column;
            align-items: center;
            gap: 20px;
        }
        .input-container input, .input-container button {
            font-size: 16px;
            padding: 12px;
            border-radius: 10px;
            border: 1px solid #FFD1DC; /* Light pink border */
            background-color: #2E2E3A; /* Darker secondary background color */
            color: #FFD1DC; /* Light pink text color */
        }
        .input-container input {
            width: 80%;
            max-width: 450px;
        }
        .input-container button {
            background-color: #FF6F9D; /* Light pinkish color */
            color: white;
            cursor: pointer;
            border: none;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease;
        }
        .input-container button:hover {
            background-color: #FF4C77; /* Darker pinkish color */
        }
        .notification-success {
            margin-top: 20px;
            padding: 15px;
            background-color: #4CAF50; /* Green background for success */
            color: white;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.3);
            font-weight: bold;
            text-align: center;
        }
        .email-preview {
            margin-top: 30px;
            padding: 20px;
            border: none;
            border-radius: 20px;
            background-color: #2E2E3A; /* Darker secondary background color */
            box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
            animation: slideIn 1s ease-in-out;
        }
        .email-preview h4 {
            margin-bottom: 15px;
            color: #FF6F9D; /* Light pinkish color */
            font-weight: bold;
        }
        .email-preview p {
            margin-bottom: 15px;
            color: #FFD1DC; /* Light pink text color */
        }
        .email-preview h3 {
            margin-bottom: 15px;
            color: #FF6F9D; /* Light pinkish color */
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        @keyframes slideIn {
            from { transform: translateY(-20px); opacity: 0; }
            to { transform: translateY(0); opacity: 1; }
        }
    </style>
    <div class="header">Notification Manager</div>
    <div class="description-box">
        <p>Fill out the form below to send a notification email about the top product category at risk of expiry.</p>
        <p>Ensure that you enter a <span>valid recipient email address</span> and provide your <span>name</span> to personalize the message.</p>
    </div>
    """, unsafe_allow_html=True)

    # Input fields for notification
    with st.form(key='notification_form'):
        recipient_email = st.text_input("Enter Recipient Email:")
        user_name = ""
        user_name = st.text_input("Enter Your Name:", "")
        send_button = st.form_submit_button("Send Notification")

        # Display progress bar and success message
        if send_button:
            if recipient_email and user_name:
                # Validate email
                try:
                    validate_email(recipient_email)

                    # Display progress bar
                    progress_bar = st.progress(0)

                    # Simulate email sending process
                    for i in range(100):
                        time.sleep(0.03)  # Simulating email sending delay
                        progress_bar.progress(i + 1)

                    # Email configuration
                    smtp_server = 'smtp.gmail.com'
                    smtp_port = 587
                    sender_email = 'winstep.noti@gmail.com'
                    sender_password = 'fxlq eipo zdmw tleo'  # Use your generated App Password

                    # Email content
                    subject = 'Urgent: Top Product Category at Risk of Expiry'
                    body = f"""
                    <html>
                    <body>
                        <p>Dear {user_name},</p>
                        <p>This is an automated notification to inform you that the top product category at risk of expiry is:</p>
                        <h3>Groceries: Cookies</h3>
                        <p>Days until expiry: <strong>5 days</strong></p>
                        <p>Please take necessary actions to address this issue.</p>
                        <p>Best,<br>Your WinStep Inventory Management Team</p>
                        <img src="https://media.licdn.com/dms/image/v2/D4E0BAQH3swEMhoL0Lg/company-logo_200_200/company-logo_200_200/0/1723750035997/stoq_team_logo?e=1732147200&v=beta&t=KTzjFhb57bSsMqfQ-O4-AZGvZBXaFlS-MgQ92GzoVjk" alt="WinStep Logo" style="width: 100px; height: auto;"/>
                    </body>
                    </html>
                    """

                    # Create the email
                    msg = MIMEMultipart('alternative')
                    msg['Subject'] = subject
                    msg['From'] = sender_email
                    msg['To'] = recipient_email
                    msg.attach(MIMEText(body, 'html'))

                    # Send the email
                    with smtplib.SMTP(smtp_server, smtp_port) as server:
                        server.starttls()
                        server.login(sender_email, sender_password)
                        server.sendmail(sender_email, recipient_email, msg.as_string())

                    # Hide progress bar and display success message
                    progress_bar.empty()
                    st.markdown('<div class="notification-success">Notification email sent successfully.</div>', unsafe_allow_html=True)

                except EmailNotValidError:
                    st.error("Invalid email address. Please check and try again.")
                except Exception as e:
                    st.error(f"Failed to send email: {e}")
            else:
                st.error("Please enter both a recipient email address and your name.")

        # Generate email preview
        email_preview_body = f"""
        <div class="email-preview">
            <h4>Email Preview:</h4>
            <p><strong>Subject:</strong> WinStep Inventory Management: Product at Risk of Expiry</p>
            <p><strong>Body:</strong></p>
            <p>Dear {user_name if user_name != "" else "<your name will appear here>"},</p>
            <p>This is an automated notification to inform you that the top product category at risk of expiry is:</p>
            <h3>Groceries: Cookies</h3>
            <p>Days until expiry: <strong>5 days</strong></p>
            <p>Please take necessary actions to address this issue.</p>
            <p>Best,<br>Your WinStep Inventory Management Team</p>
        </div>
        """

        # Display email preview
        st.markdown(email_preview_body, unsafe_allow_html=True)

# Sample call to the function
# send_email_notification('Electronics', 'recipient@example.com')

##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################################################################################################################################################################################

# Main function to run the Streamlit app
def main():
    #st.title("Shelf-Life Analysis Application")
    key_path_dict = {
        "type": "service_account",
        "project_id": "winstep-16ca4",
        "private_key_id": "e21fa56abf951b80ec13ef31b9760fee89f2c327",
        "private_key": "-----BEGIN PRIVATE KEY-----\nMIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQC/CMWBasSZpF91\no/N354giNKF4Dp3sW/39CjnOjUsbIrrOLnuRXzC0Lfb27qsGMpjLfYEuZij/4vIU\nx3Mk47SjsyzCE6hOV2ykbbTK+NPCTFZ5ytFK+MWxfpRPYWd30u1+QE2ea0MFtSFx\nRR4x4Aot/MLrzCAOMfJPeV4T1ld+kiibB32PYJE28Mm5XBQMzfNfD1E1a5wAOtCO\nMoqLXXWmWie0pX4Z0O/iL9RuYs7bSDhNV8Jvrv9wtAvtxrMGtR6f2Q2JumXLgGgc\nOYox2FAOheNtduNojkN7gzl1hB/McwWNWxfR9wXRRVaR4efUeCzrgk2xk4vZkxrs\nDpaIAzaVAgMBAAECggEALIeSZlbpafj/SXBEpewB9xs2kkIx/LD61PuHuwaDfdlM\nGxiJtDqwoyddvHSoyAKOTEy+in7EytTvYfmV9QDhEBGJetnTLyPAztlFvdRfpBhg\nRfaJb8TqbDPZxWEqmatAsd+yWB2fm1p756fZYH3dUZfsJcPIqxZoaa8cR1p8vaNU\nEy4dck4kx4bk+gPA8TdxwHCfm2e5f7CB0+WwHa9BEdLE9be7DgRL1ELFv8BOoBYM\nDLe+rUmwbVaZP2OWqzAt6X/nAsipTXxLQPLdKxokfdAiJfDtIDEZRVGj0kk0mH85\nX9hzzLBsraXNoY/0vjwQTUAGllX7bCu6Mc39vG6D9wKBgQDi1zQEI8LuV+FFb/fh\n2kdvRqsiC+qU0q9Ym7vX3zlqJPvVzR59VLtr0E9LmOzFfGbkleN7s3neFgYv0wdC\nNzrH5IwUvqjTW+v0NzDrAv9FkAR+NCqp1kWAVV3zgHMWhe2k8xd3FiJAoABeBB4n\nLbV3xZwF6hBF715er5FvYA48dwKBgQDXl0OU5WSXY6/hT4ENvcjNR0PQ2VRNnLuD\nRMLC3vX9hy+i7aycepcDnLrwNCZfTEBED3gsm/npwobN61N6UNvXmIowEvCXXlnu\nOwyf/OmwMhio90EQJxKaEZf8d6SBrTcyN0dppgit/53Q36ZnkoMWDY2rBWow1uPh\nn1CPUtpEUwKBgQDMUzwvXmb/eXkYqrqFXbBqsyUDDejHFN+M2PpigFefHKEa/CAy\nlFgdzQ0f8yeS23NzAvBdRFTJjt0TxuoK4uS3mU30gahgebQXzn7psVFuv0LMywCC\n6ta/uiVeaJ1B9HES20SPqAhCXdz20o62i52hvQXE7giqdepzL4G46LTqEQKBgCF/\ngGG3Tuzy8VYZ61x+O6AhzZi63A1/J+eanIR47lHpWm5/bY2WwrYt+SHviHLQP0AU\nA0EzLx6yOg3u3baor7ANJJOZrcZnQ6PviuOlAY5+CjTezj47Q/mqeCojUO1RQ71K\nt47j3H9ks1nMFmgLbNDVZEjJe5mBGkFpZrQOVJm/AoGAEeDE7kEihF02FMcvKYwK\ni+rINaFK7mjGqZANugIv/ZeoOxtyTVbygtYUtwSTL/STxSiopCn0iBdK/g+rb8Jo\ncbAf5G26Og9T5M9SPkO3U4EjvfO1iHrEooiRoSfHgUjABFo2stKpcrkn4EoLr1Hn\n+jTtDP9qw47bBODXfLSnusg=\n-----END PRIVATE KEY-----\n",
        "client_email": "firebase-adminsdk-7ye8r@winstep-16ca4.iam.gserviceaccount.com",
        "client_id": "104601567201728476313",
        "auth_uri": "https://accounts.google.com/o/oauth2/auth",
        "token_uri": "https://oauth2.googleapis.com/token",
        "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
        "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/firebase-adminsdk-7ye8r%40winstep-16ca4.iam.gserviceaccount.com",
        "universe_domain": "googleapis.com"
    }

    if not firebase_admin._apps:
        cred = credentials.Certificate(key_path_dict)
        firebase_admin.initialize_app(cred)

    db = firestore.client()


    #st.sidebar.title("Navigation")
    #page = st.sidebar.radio("Choose a page", ["Home Page", "Capture & Analyze", "Review Data"])


    with st.sidebar:
      # Display the styled sidebar content
      st.markdown("""
          <style>
          .sidebar-subtext {
              font-size: 16px;
              color: #FFFFFF; /* White color */
              text-align: center;
              margin-bottom: 20px;
          }
          .sidebar-header {
              font-size: 24px;
              font-weight: bold;
              color: #FF69B4; /* Pink color */
              text-align: center;
              margin-bottom: 10px; /* Reduced margin for separator */
          }
          .separator {
              border-top: 2px solid #FF69B4; /* Pink separator */
              margin-bottom: 20px;
          }
          .current-page {
              font-size: 18px;
              font-weight: bold;
              color: #FF69B4; /* Pink color for page */
              text-align: center;
              margin-top: 10px;
          }
          .stOptionMenu {
              margin-top: 20px;
          }
          </style>
          <div class="sidebar-header">Welcome to <span style="color: #FFFFFF;">WinStep</span>!</div>
          <div class="separator"></div>
          <div class="sidebar-subtext">Please select a navigation page from the options below.</div>
      """, unsafe_allow_html=True)

      # Compute the current page selection
      page = option_menu(
          "Navigation",
          ["Dashboard", 'Capture & Analyze', 'Review Data', 'Risk Analysis', 'Notification Manager', '---'],
          icons=['house', 'camera', 'code-slash', 'graph-up-arrow', 'envelope'],
          menu_icon="cast",
          default_index=0
      )




    #st.logo("https://i.ibb.co/FszJXsx/Infosys-logo-2-optimized.png")
    left_co, cent_co,last_co = st.columns(3)
    with cent_co:
        st.image("https://media.licdn.com/dms/image/v2/D4E0BAQH3swEMhoL0Lg/company-logo_200_200/company-logo_200_200/0/1723750035997/stoq_team_logo?e=1732147200&v=beta&t=KTzjFhb57bSsMqfQ-O4-AZGvZBXaFlS-MgQ92GzoVjk", use_column_width=False)


    if page == "Dashboard":
        display_home_page()

    elif page == "Review Data":
        categories = fetch_all_categories(db)
        category = st.sidebar.selectbox("Select a category", categories, index = 3)

        if category:
            st.session_state["category"] = category
            st.header(f"Inventory Review: {category}")
            data = fetch_data_from_firebase(db, category)

            if data:
                df = pd.DataFrame(data)
                st.write("### Inventory Data", df)
                col1, col2, col3 = st.columns(3)
                for idx, item in enumerate(data):
                    col = [col1, col2, col3][idx % 3]
                    if col.button(f"{item['Product name']}", key=f"{item['Product name']}_{idx}"):
                        st.session_state["selected_product"] = item
                        st.session_state["page"] = "Product Details"
                    col.image(item["Image URL"], caption=item["Product name"], use_column_width=True, width=300)
            else:
                st.write(f"No data found for category: {category}")

    if st.session_state.get("page") == "Product Details":
        if "selected_product" in st.session_state:
            product = st.session_state["selected_product"]
            st.header(product["Product name"])
            st.image(product["Image URL"], use_column_width=True)
            st.json(product)
            if st.button("Back"):
                st.session_state["page"] = "Review Data"
        else:
            st.write("No product selected.")

    elif page == "Capture & Analyze":
        #db = initialize_firebase()  # Initialize Firebase here to ensure db is available
        # Get the current date
        date_time = get_current_date()
        if date_time:
            img = upload_and_display_image()
            if img:
                st.session_state["img"] = img
                val = get_image_value(img)
                if val:
                    imgbb_api_key = "7fbd8710350bbf47d40088ff68ce1607"
                    image_url = upload_image(val, imgbb_api_key)
                    if image_url:
                        response_content = analyze_uploaded_image(image_url, date_time)
                        return_tuple = (image_url, response_content)
                        display_analysis_results(response_content)

                        # Extract product category before popping General Info
                        product_category = response_content["General Info"]["Product category"]

                        general_info = response_content.pop("General Info")
                        general_info["Image URL"] = image_url

                        collection_ref = db.collection(product_category)
                        collection_ref.add(general_info)
    elif page == "Risk Analysis":
        display_expiry_risk_analysis()
    elif page == "Notification Manager":
        display_notification_manager()

if __name__ == "__main__":
    main()

