import pandas as pd
import matplotlib.pyplot as plt 
import uuid
from datetime import datetime, timedelta
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier


class ReturnRatePredictor:
    def __init__(self, csv_file):
        self.censored_sales_data = self.load_data(csv_file)
        self.check_file()
    
    def check_file(self):
        upload = self.censored_sales_data

        required_columns = [
            'id',
            'customer_id', 
            'exclude', 
            'financial_status', 
            'fulfillment_status', 
            'accepts_marketing', 
            'currency', 
            'subtotal', 
            'shipping', 
            'discount_code', 
            'refunded_amount', 
            'discount_amount', 
            'shipping_method', 
            'created_at', 
            'lineitem_quantity', 
            'size', 
            'lineitem_sku', 
            'billing_zip', 
            'billing_country', 
            'payment_method', 
            'card_type', 
            'transactions', 
            'bought_multiple_diff_size'
        ]

        file_columns = upload.columns.tolist()

        # Check for missing and extra columns
        missing_columns = [col for col in required_columns if col not in file_columns]
        extra_columns = [col for col in file_columns if col not in required_columns]

        if missing_columns:
            print(f'Missing columns: {", ".join(missing_columns)}. Fix CSV file and try again.')
        elif extra_columns:
            print(f'Extra columns: {", ".join(extra_columns)}. Fix CSV file and try again.')  # Corrected to print extra_columns
        else:
            print('CSV file contains all necessary information to proceed. Analysis can commence.')
            self.start_analysis()  # Assuming this method exists and starts the analysis

    def load_data(self, csv_file):
        try:
            return pd.read_csv(csv_file)
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            return None
        
    def start_analysis(self):
        user_input = input("Would you like to start the analysis? (y/n): ").strip().lower()
        
        if user_input == 'y':
            print('Cleaning data...')
            self.clean_data()
            print('Calculating additional features...')
            self.calculate_additional_features()
            print('Preparing data for analysis...')
            self.data_preparation()
            print("Analysis started.")
            self.categorical_predictor()
            print('Analysis completed successfully! Creating report...')
            self.generate_summary_report()
            print('Report completed.')
        else:
            print("Analysis cancelled.")
            return

    # remove rows that are demo or test orders from source, contains NAs, and ensure boolean values
    def clean_data(self):
        self.censored_sales_data['created_at'] = pd.to_datetime(self.censored_sales_data['created_at'], format='%m/%d/%y %H:%M', errors='coerce')
        self.censored_sales_data = self.censored_sales_data[self.censored_sales_data['exclude'] == False]
        self.censored_sales_data = self.censored_sales_data[self.censored_sales_data['fulfillment_status'] == 'fulfilled']
        self.censored_sales_data.dropna(inplace=True)
        self.censored_sales_data['id'] = self.censored_sales_data['id'].apply(str)
        self.censored_sales_data['customer_id'] = self.censored_sales_data['customer_id'].apply(str)
        self.censored_sales_data['size'] = self.censored_sales_data['size'].apply(str)
        self.censored_sales_data['accepts_marketing'] = self.censored_sales_data['accepts_marketing'].replace({'no': False, 'yes': True}).astype(bool)
    
    def get_clothing_category(self, size):
        categories = {
            'hat': ['1'],
            'dress_shirt': ['38','39','40','41','42','43'],
            'pants': ['32','33','34','35','36'],
            'suit': ['44','46','48','50','52','54','56','58','60','92','96','D100','D104','D108','D112','D116','D120'],
            'shirt': ['XXS', 'XS', 'S', 'M', 'L', 'XL', 'XXL']
            }
        
        for category, sizes in categories.items():
            if size in sizes:
                return category
            return 'Unknown'
    
    def aggregated_orders(self):
        order_aggregate = self.censored_sales_data.groupby('id')[['subtotal', 'shipping', 'lineitem_quantity']].sum().reset_index()
        return order_aggregate

    # add more features
    def calculate_additional_features(self):
        # create line item identifier
        self.censored_sales_data['line_item_id'] = [uuid.uuid4() for _ in range(len(self.censored_sales_data))]

        # create features related to time of order
        self.censored_sales_data['hour_of_day'] = self.censored_sales_data['created_at'].dt.hour
        self.censored_sales_data['day_of_week'] = self.censored_sales_data['created_at'].dt.dayofweek
        self.censored_sales_data['day_of_month'] = self.censored_sales_data['created_at'].dt.day
        self.censored_sales_data['month_of_year'] = self.censored_sales_data['created_at'].dt.month
        self.censored_sales_data['is_sunday'] = self.censored_sales_data['created_at'].dt.weekday == 6
        self.censored_sales_data['is_weekend'] = (self.censored_sales_data['created_at'].dt.weekday == 6) | (self.censored_sales_data['created_at'].dt.weekday == 5) 
        
        # add clothing category
        self.censored_sales_data['clothing_category'] = self.censored_sales_data['size'].apply(self.get_clothing_category)
        
        # calculate if returning customer
        purchase_count = self.censored_sales_data.groupby('customer_id')['created_at'].nunique()
        self.censored_sales_data['first_time_shopper'] = self.censored_sales_data['customer_id'].map(lambda x: purchase_count.get(x, 0) == 1)

        # calculate order values (aggregated on order-level)
        order_aggregated = self.aggregated_orders()
        self.censored_sales_data = pd.merge(self.censored_sales_data, order_aggregated, on='id', how='left', suffixes=('', '_agg'))

    # prepare data for ensemble model
    def data_preparation(self):
        # exclude orders made in the last 14 days as these can still be returned per webshop policy
        self.censored_sales_data = self.censored_sales_data[self.censored_sales_data['created_at'] < datetime.today() - timedelta(days=14)]

        # convert columns to numeric values
        categories = [
            'financial_status', 
            'fulfillment_status',
            'currency',
            'shipping_method',
            'size', 
            'lineitem_sku',
            'billing_zip',
            'billing_country', 
            'payment_method',
            'card_type',
            'clothing_category'
            ]
        
        for category in categories:
            label_encoder = LabelEncoder()
            self.censored_sales_data[category] = label_encoder.fit_transform(self.censored_sales_data[category])

        # convert booleans to 1/0
        booleans = [
            'accepts_marketing',
            'discount_code',
            'bought_multiple_diff_size',
            'is_sunday',
            'is_weekend',
            'first_time_shopper'
            ]
        
        for boolean in booleans:
            self.censored_sales_data[boolean] = self.censored_sales_data[boolean].astype(int)

        # exclude irrelevant features
        # self.censored_sales_data = self.censored_sales_data.drop(['id', 'customer_id', 'exclude', 'fulfillment_status', 'created_at', 'transactions'], axis=1)
        
        # scale necessary features
        scaler = StandardScaler()
        
        features_to_scale = [
            'subtotal',
            'shipping',
            'subtotal_agg',
            'shipping_agg',
            'refunded_amount',
            'lineitem_quantity'
        ]

        self.censored_sales_data[features_to_scale] = scaler.fit_transform(self.censored_sales_data[features_to_scale])

        # Use only relevant variable (remove refunded-related features because they will not be known when predicting new)
        self.X = self.censored_sales_data[[
            'customer_id',
            'accepts_marketing',
            'currency',
            'subtotal',
            'discount_code',
            'shipping_method',
            'lineitem_quantity',
            'size',
            'lineitem_sku',
            'billing_zip',
            'billing_country',
            'payment_method', 
            'card_type',
            'bought_multiple_diff_size',
            'hour_of_day', 
            'day_of_week',
            'day_of_month',
            'month_of_year',
            'is_sunday',
            'is_weekend',
            'first_time_shopper',
            'clothing_category',
            'subtotal_agg',
            'shipping_agg',
            'lineitem_quantity_agg'
        ]]

        # categorical status of refunded or not as target variable
        self.y_cat = self.censored_sales_data['financial_status']
        self.y_user = self.censored_sales_data['customer_id']

        return self.X, self.y_cat, self.y_user # define two different target variables to be used for categorical and regression prediction

    def categorical_predictor(self):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y_cat, test_size=0.25, random_state=42)

        knn = KNeighborsClassifier(n_neighbors=5)
        rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        lr = LogisticRegression()
        
        hard_voting = VotingClassifier(estimators=[
            ('knn', knn),
            ('lr', lr),
            ('rf', rf)
        ], voting='hard')
        
        soft_voting = VotingClassifier(estimators=[
            ('knn', knn),
            ('lr', lr),
            ('rf', rf)
        ], voting='soft')
        
        models = {
            'kNN': knn,
            'Logistic Regression': lr,
            'Random Forest': rf,
            'Hard Voting': hard_voting,
            'Soft Voting': soft_voting
        }

        best_model_name = None
        best_accuracy = 0
        best_model = None
        best_predictions = None
        best_true_rows = None
        
        # for each model, perform regular ML stuff
        for model_name, model in models.items():
            print(f'Training {model_name}...')
            model.fit(X_train, y_train)
            print(f'Testing {model_name}...')
            predictions = model.predict(X_test)
            
            accuracy = accuracy_score(y_test, predictions)
            print(f'{model_name} Accuracy: {accuracy:.4f}')

            # return best performing model
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = model_name 
                best_model = model
                best_predictions = predictions
                best_true_rows = X_test[predictions == 1]

        if best_accuracy <= 0.5:
            self.message = '\nWarning! None of the models performed better than a coin-toss. Results ought to be disregarded.'
        else:
            self.message = f'\n{best_model_name} achieved a satisfactory accuracy of {best_accuracy:.4f} when predicting test data. Results can be taken into consideration for business decisions. \nFull Classification Report for this model: \n {classification_report(y_test, predictions)}'

        users_list = best_true_rows["customer_id"].unique().tolist()
        self.users = pd.DataFrame(users_list, columns=['Customer_ID'])
        
        # generate what features have an impact on the prediction from best performer random forest
        feature_importances = rf.feature_importances_
        features = self.X.columns

        sorted_idx = feature_importances.argsort()[::-1]
        sorted_feature_importances = feature_importances[sorted_idx]
        sorted_features = features[sorted_idx]
        
        plt.figure(figsize=(10, 6))
        plt.barh(sorted_features, sorted_feature_importances)
        plt.xlabel('Feature Importance')
        plt.title('Random Forest - Feature Importance')
        plt.gca().invert_yaxis() 
        plt.savefig('ai-development/Project/project_output/feature_importance.png')
        plt.close()
    
    def generate_summary_report(self):
        report = f"""
Refunded Line Items Summary Report
=====================================
{self.message}
The following list of users were correctly predicted to return their items. Evaluate necessary strategy:\n 
{self.users.to_string(index=False)}

        """
        with open('ai-development/Project/project_output/summary_report.txt', 'w') as f:
            f.write(report)

def main():
    predictor = ReturnRatePredictor('ai-development/Project/data_import/censored_sales_data.csv')
    
    predictor

if __name__ == "__main__":
    main()

