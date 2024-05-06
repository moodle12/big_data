from flask import Flask, render_template, request
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

app = Flask(__name__)

# Load the trained model
spark = SparkSession.builder \
    .appName("ClassificationModel") \
    .getOrCreate()

model = PipelineModel.load("path_to_your_model")

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
        # Add more features as needed
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
