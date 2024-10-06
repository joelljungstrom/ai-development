import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from datetime import datetime, timedelta

class WeatherAnalysis:
    def __init__(self, csv_file):
        self.df = self.load_csv(csv_file)
        self.clean_data()
        self.calculate_monthly_avg_temp()
        self.calculate_additional_features()
    
    def load_csv(self, csv_file):
        df = pd.read_csv(csv_file, index_col='Date', parse_dates=True)
        df.sort_index(inplace=True)
        return df 
    
    def clean_data(self):
        self.df.dropna(inplace=True)
        self.df.drop_duplicates(inplace=True)
    
    def calculate_monthly_avg_temp(self):
        #self.df['Date'] = pd.to_datetime(self.df['Date'])
        temperature_per_month = self.df.groupby(self.df.index.to_period('M'))['Temperature'].mean().reset_index()
        return temperature_per_month
    
    def calculate_additional_features(self):
        self.df['3DA_Temperature'] = self.df['Temperature'].rolling(window=3).mean()
        self.df['7DA_Temperature'] = self.df['Temperature'].rolling(window=7).mean()

    def plot_temperature(self):
        plt.figure(figsize=(12,6))
        plt.plot(self.df.index, self.df['Temperature'], label = 'Temperature (C°)')
        plt.title('Daily Temperature')
        plt.xlabel('Date')
        plt.ylabel('C°')
        plt.legend()
        plt.savefig('ai-development/Exercises/week3/Weather/generated_files/temperatures.png')
        plt.close()
    
    def plot_precipitation(self):
        plt.figure(figsize=(12,6))
        plt.bar(self.df.index.to_period('M').astype(str), self.df['Precipitation'], label = 'Precipitation (mm)')
        plt.title('Daily Precipitation')
        plt.xlabel('Month')
        plt.ylabel('Precipitation (mm)')
        plt.legend()
        plt.savefig('ai-development/Exercises/week3/Weather/generated_files/precipitation.png')
        plt.close()
    
    def plot_temp_humidity(self):
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.scatter(self.df.index, self.df['Temperature'], label = 'Temperature', color='r')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Temperature (C°)', color='r')
        ax1.tick_params(axis='y', labelcolor='r')

        ax2 = ax1.twinx()
        ax2.scatter(self.df.index, self.df['Humidity'], color='b', label='Humidity')
        ax2.set_ylabel('Humidity (%)', color='b')
        ax2.tick_params(axis='y', labelcolor='b')

        plt.title('Temperature and Humidity')
        plt.legend()
        plt.savefig('ai-development/Exercises/week3/Weather/generated_files/temp_and_humidity.png')
        plt.close()
    
    def plot_heatmap(self):
        correlation_matrix = self.df.loc[:, self.df.columns != 'Date'].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Weather Data Correlation Heatmap')
        plt.legend()
        plt.savefig('ai-development/Exercises/week3/Weather/generated_files/heatmap.png')
        plt.close()

    def predict_temperature(self):
        self.df['Predicted_Next_Day_Temperature'] = (self.df['3DA_Temperature'].shift(1) + self.df['7DA_Temperature'].shift(1)) / 2
        
        prediction_df = pd.DataFrame({
            'Date': self.df.index,
            'Predicted_Temperature': self.df['Predicted_Next_Day_Temperature']
        }).set_index('Date')

        return prediction_df
        # Take the average between 3- and 7-day rolling average to predict the next-day temperature.
    
    def plot_predicted_temperature(self, prediction_df):
        plt.figure(figsize=(12,6))
        plt.plot(self.df.index, self.df['Temperature'], label='Actual Temperature')
        plt.plot(prediction_df.index, prediction_df['Predicted_Temperature'], label='Predicted Temperature', linestyle='--')
        plt.title('Actual vs Predicted Temperature')
        plt.xlabel('Date')
        plt.ylabel('Temperature (C°)')
        plt.legend()
        plt.savefig('ai-development/Exercises/week3/Weather/generated_files/temperature_prediction.png')
        plt.close()
    
    def statistics_overview(self):
        filtered_df = self.df.drop(columns=['3DA_Temperature', '7DA_Temperature', 'Predicted_Next_Day_Temperature'])
        summary_stats = filtered_df.describe().T
        
        summary_stats['median'] = filtered_df.median()
        summary_stats['Q1'] = filtered_df.quantile(0.25)
        summary_stats['Q3'] = filtered_df.quantile(0.75)
        summary_stats['std'] = filtered_df.std()

        summary_stats = summary_stats[['mean', 'median', 'Q1', 'Q3', 'std', 'min', 'max']]
        summary_stats.columns = ['Mean', 'Median', '1st Quartile (Q1)', '3rd Quartile (Q3)', 'Standard Deviation', 'Min', 'Max']

        summary_stats = summary_stats.round(2)

        return summary_stats
    
    def evaluate_prediction(self, prediction_df):
        mse = np.mean((self.df['Temperature'] - prediction_df['Predicted_Temperature']) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(self.df['Temperature'] - prediction_df['Predicted_Temperature'])

        return rmse, mae

    def generate_summary(self, summary_stats, evaluation):
        rmse, mae = evaluation
        report = f"""
Stock Analysis Summary Report
=============================
Period: {self.df.index[0]} to {self.df.index[-1]}

Key Statistics:
-----------------------------
Overview of statistical metrics:
{summary_stats}

Prediction Result:
-----------------------------
Root Mean Squared Error of predicted temperature: {rmse:.2f}
Mean Absolut Error of prediction of predicted temperature: {mae:.2f}
        """
        with open('ai-development/Exercises/week3/Weather/generated_files/summary_report_weather.txt', 'w') as f:
            f.write(report)
    
def main():
    analyzer = WeatherAnalysis('ai-development/Exercises/week3/Weather/weather_data.csv')
    
    print("Generating visualizations...")
    analyzer.plot_temperature()
    analyzer.plot_precipitation()
    analyzer.plot_temp_humidity()
    analyzer.plot_heatmap()

    print("Predicting next-day temperature and evaluating results...")
    prediction = analyzer.predict_temperature()
    analyzer.plot_predicted_temperature(prediction)
    evaluation = analyzer.evaluate_prediction(prediction)
    
    print("Generating report...")
    summary = analyzer.statistics_overview()
    analyzer.generate_summary(summary, evaluation)

if __name__ == "__main__":
    main()
