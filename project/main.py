from flask import Flask, render_template, request
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

app = Flask(__name__)

# Load the trained model
spark = SparkSession.builder \
    .appName("ClassificationModel") \
    .getOrCreate()

model = PipelineModel.load("./gbt_model")

# Define route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Define route for form submission
@app.route('/predict', methods=['POST'])
def predict():
    # Get the form data
    features = {
        'funding_total_usd': request.form['funding_total_usd'],
        'funding_rounds': request.form['funding_rounds'],
        'country_code': request.form['country_code'],
        'city': request.form['city'],
        'founded_year': request.form['founded_year'],
        'category_final': request.form['category_final'],
        'total_raised_usd': request.form['total_raised_usd']
    }

    # Create a DataFrame from the form data
    df = spark.createDataFrame([features])

    # Make predictions using the model
    predictions = model.transform(df)

    # Get the prediction result
    prediction = predictions.select('prediction').collect()[0][0]

    # Display the prediction result
    result = "Acquisition" if prediction == 1 else "Not Acquisition"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
