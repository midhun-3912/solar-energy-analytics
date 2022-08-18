import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate, ShuffleSplit
from matplotlib import pyplot as plt
import urllib as urllib
import altair as alt


st.set_page_config(layout="wide")
st.title("Solar Energy Analytics")
st.write("""Welcome to the solar panel power prediction app""")

# sidebar
st.sidebar.write("Select Date Range")
location = st.sidebar.text_input("Location",
    value="560001",
    max_chars=6,
    type="default",
    help="Please enter the zip code of the location")
start_date = st.sidebar.date_input("Start Date")
end_date = st.sidebar.date_input("End Date",
    value=start_date,
    min_value=start_date)

# API to get wheather forecast data for the selected location and time period
url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{0}/{1}/{2}?unitGroup=metric&include=hours&key=VQ9KXEAE5GFJ9FM9658HDWVA6&contentType=csv".format(location, start_date, end_date)

def get_dataset():
    data = pd.read_csv("Measured DataSet.csv", header = 0)
    return data
 
def get_wheather_forecast_data(url):
    try:
        CSVText = pd.read_csv(url)
    except urllib.error.HTTPError  as e:
        ErrorInfo= e.read().decode()
        CSVText = "Error code: {0}, {1}".format(e.code, ErrorInfo)
    except  urllib.error.URLError as e:
        ErrorInfo= e.read().decode()
        CSVText = "Error code: {0}, {1}".format(e.code, ErrorInfo)
    return CSVText

# fit the curve using Linear Regression
data = get_dataset()
X = data[["solarradiation"]]
y = data["Measured Power (mW)"]
model = LinearRegression()
model.fit(X, y)

# get the solar radiation data from the web API
forecast_data = get_wheather_forecast_data(url)
rad_forecast = forecast_data[["solarradiation"]]

# estimate the power for corresponds to the solar radiation data
y_estimated = model.predict(rad_forecast)
y_estimated = pd.DataFrame(y_estimated, columns = ['Estimated Power (mW)'])

# display the chart and data table
st.write("""### Estimated Power Chart""")

table_display_1 = pd.concat([forecast_data[["datetime", "temp", "conditions", "solarradiation"]], y_estimated], axis=1)
table_display_1.rename(columns = {'solarradiation':'Solar Radiation (W/m2)'}, inplace = True)
table_display_1.rename(columns = {'datetime':'Date and Time'}, inplace = True)
table_display_1.rename(columns = {'temp':'Ambient Temperature (Deg C)'}, inplace = True)
table_display_1.rename(columns = {'conditions':'Conditions'}, inplace = True)


col01,col02 = st.columns(2)
avg_est_power = round(table_display_1["Estimated Power (mW)"].mean(), 2)
col01.metric("Average Estimated Power (mW): ", avg_est_power)

chart_1 = alt.Chart(table_display_1).mark_bar(opacity=1).encode(x='Date and Time',
    y='Solar Radiation (W/m2)')
chart_2 = alt.Chart(table_display_1).mark_bar(opacity=0.6).encode(x='Date and Time', 
    y='Estimated Power (mW)',
    tooltip=['Date and Time','Solar Radiation (W/m2)','Estimated Power (mW)'])
c = alt.layer(chart_1, chart_2)
st.altair_chart(c, use_container_width=True)

st.write("**Table 1: Detailed View of Hourly Solar Radiation (W/m2) Data From: {0} To: {1} and the Estimated Solar Power (mW)**".format(start_date, end_date))
st.dataframe(table_display_1.style.highlight_max('Estimated Power (mW)'), width=None)

# model evaluation
st.write("""### Model Evaluation""")
st.write("Shape of Training Dataset", data.shape)
col03,col04 = st.columns(2)
accuracy_model = round(model.score(X,y), 3)
col03.metric("Accuracy of the model: ", accuracy_model)

y_pred = model.predict(X)
y_pred = pd.DataFrame(y_pred, columns = ['Predicted Power (mW)'])
table_display_2 = pd.concat([data[["solarradiation", "Measured Power (mW)"]], y_pred.reindex(data.index)], axis=1)
table_display_2.rename(columns = {'solarradiation':'Solar Radiation'}, inplace = True)

st.write("**Table 2: Measured Solar Radiation (W/m2), Power (mW) generated from the Solar Panel and Power (mW) predicted using Linear Regression Model**")
col05,col06 = st.columns(2)
col05.dataframe(table_display_2, width=None)

fig = plt.figure()
plt.scatter(X, y, color="red", alpha=0.5)
plt.plot(X, y_pred)
plt.title("Linear Regression Model")
plt.xlabel("Solar Radiation (W/m2)")
plt.ylabel("Solar Power (mW)")
col06.pyplot(fig)

# evaluate the model using train and test split method
st.write("""
         ##### **Split the data into Train and Test sets**
         """)
size = st.slider("Size of the test set", min_value=0.05, max_value=1.00, step=.05, value=.25)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=size, random_state=10)
model.fit(X_train, y_train)
test_score = model.score(X_test, y_test)
train_score = model.score(X_train, y_train)

st.write("The correlation coefficient for training data is: ", format(train_score, ".3f"))
st.write("The correlation coefficient for test data is: ", format(test_score, ".3f"))

# evaluate the model using cross validation
st.write("""
         ##### **Evaluate the model using Cross-Validation**
         """)
cv = ShuffleSplit(n_splits=40, test_size=size, random_state=0)
cv_results = cross_validate(model, X, y, cv=cv, return_train_score = True)
st.write("The correlation coefficient for training data is: ", format(cv_results['train_score'].mean(), ".3f"))
st.write("The correlation coefficient for test data is: ", format(cv_results['test_score'].mean(), ".3f"))

# display the data obtained from the web API
st.write("""
         ### Collected Data from Visual Crossing Web Services
""")
st.write(url)
st.write("**Table 3: Forecast Data at Location: {0} From: {1} To: {2}**".format(location, start_date, end_date))
st.write(forecast_data)

# end