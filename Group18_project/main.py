from flask import Flask, render_template, request
from pyspark.ml.pipeline import PipelineModel
from pyspark.sql import SparkSession

app = Flask(__name__)

# Load the trained model
spark = SparkSession.builder \
    .appName("ClassificationModel") \
    .getOrCreate()

# Specify the path to the saved model folder
mPath = "./gbt_model"

# Load the model
model = PipelineModel.load(mPath)

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
    sample_row = spark.createDataFrame([features])

    # Make prediction on the sample row
    prediction = model.transform(sample_row)

    # Show the prediction result
    prediction.select("features", "prediction").show()

    # Get the prediction result
    prediction_value = prediction.select('prediction').collect()[0][0]

    # Display the prediction result
    result = "Acquisition" if prediction_value == 1 else "Not Acquisition"

    return render_template('result.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
