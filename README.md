💉 COVID-19 Vaccine Forecasting

📖 Introduction
This project is a COVID-19 Vaccine Forecasting System built using Python, Pandas, Scikit-learn, Matplotlib, and Seaborn.
It uses a Decision Tree Regression model to forecast the number of vaccines manufactured based on historical data. The project also visualizes trends across different locations and vaccine manufacturers using interactive charts.


💻 Technologies
Python – Versatile, easy-to-use programming language.
Pandas – For data manipulation and preprocessing.
NumPy – For efficient numerical computations.
Scikit-learn – For machine learning and forecasting.
Matplotlib & Seaborn – For data visualization and trend analysis.


✨ Features
📊 Load and clean historical COVID-19 vaccine data.
🏭 Visualize total vaccinations by manufacturer.
🌍 Location-wise analysis of vaccine distribution.
🔁 Transform time series data into supervised learning format.
📈 Forecast upcoming vaccine production using machine learning.
🧪 Evaluate model accuracy using RMSE.
📉 Plot Actual vs Forecasted vaccine counts.


🧱 Architecture
Follows a standard data science pipeline:
Data Collection – Vaccine data from a CSV file.
Data Preprocessing – Pivot tables and date formatting.
Visualization – Manufacturer & location-based visualizations.
Modeling – Decision Tree Regressor for prediction.
Evaluation – Forecast analysis using plots and RMSE.


📦 Modules
Data Cleaning Module – Reads CSV, handles dates, structures data.
Visualization Module – Plots for manufacturer and location analysis.
Forecasting Module – Builds and tests a Decision Tree Regression model.
Evaluation Module – Compares predictions vs actuals, calculates RMSE.


👤 Users
Perfect for:
📚 Students learning time series forecasting.
📈 Data Analysts interested in real-world COVID-19 data.
🧪 Researchers exploring vaccine manufacturing trends.
🚀 Getting Started


✅ Prerequisites
Make sure the following tools and libraries are installed:
Python 3.8+
pip
Python libraries:
bash
pip install pandas numpy matplotlib seaborn scikit-learn


🔧 Steps to Run the Application
1️⃣ Clone the Repository
bash
git clone https://github.com/yourusername/covid19-vaccine-forecast.git
cd covid19-vaccine-forecast

2️⃣ Add the Dataset
Place the vaccinations.csv file inside a folder named Dataset/.
Example path:
bash
Dataset/vaccinations.csv

3️⃣ Run the Forecast Script
bash
python Forcast.py
📍 Output
📈 Graphs showing vaccine manufacturing trends.

🔮 Forecasted values for vaccine distribution.

📊 RMSE score for prediction accuracy.
