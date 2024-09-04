from flask import Flask, render_template, request, flash
from flask_wtf import FlaskForm
import io
import base64
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from Stock_Price_Predictor_Website.API_call import response
import pandas as pd
import tensorflow
import keras
import matplotlib
import os
matplotlib.use('Agg')  # Use a non-GUI backend


app = Flask(__name__)

# Set the secret key to a random value or use an environment variable
# Or you can use a fixed string: 'your_secret_key'
app.secret_key = os.urandom(24)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/", methods=["POST", "GET"])
def predictor():
    status = False

    if request.method == "POST":
        ticker = request.form.get("stock_name")
        status = True

    if status == True:

        # Recieve and format data for use
        data = response(ticker.upper())

        if data.status_code != 200 or not data.text.strip():
            flash(
                f"No data found for the ticker '{ticker}'. Please try another ticker.")
            return render_template("index.html")

        df = pd.read_csv(io.StringIO(data.text))
        df = df.iloc[::-1]

        if df.empty or 'close' not in df.columns:
            flash(
                f"No valid stock data available for '{ticker}'. Please try another ticker.")
            return render_template("index.html")

        close_data = df.filter(['close'])  # Filter only closing data set
        dataset = close_data.values  # Put closing dataset into a 2D array
        # Get only 89% of data to be used as training
        training = int(np.ceil(len(dataset) * 0.97))

        try:
            # Scale range for model
            scaler = MinMaxScaler(feature_range=(0, 1))
        except Exception as e:
            flash(f"Error processing data: {e}")
            return render_template("index.html")

        scaled_data = scaler.fit_transform(
            dataset)  # Scale the close value dataset

        train_data = scaled_data[0:int(training), :]

        x_train = []  # Input train
        y_train = []  # Output train

        # Sliding window approach to train data
        for i in range(500, len(train_data)):
            x_train.append(train_data[i-500:i, 0])
            y_train.append(train_data[i, 0])

        # Convert both training arrays into a numpy one
        x_train, y_train = np.array(x_train), np.array(y_train)
        # Convert input data in a 3D shape for the LSTM
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        model = keras.models.Sequential()  # Creates a empty sequenital
        model.add(keras.layers.Input(shape=(x_train.shape[1], 1)))
        # Creates a LSTM layer with 8 units
        model.add(keras.layers.LSTM(units=32, return_sequences=True))
        model.add(keras.layers.LSTM(units=32))
        # Creates a dense layer where everything is fully connected
        model.add(keras.layers.Dense(16))
        # Randomly dropsout neurons to prevent overfitting
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1))

        # Parameters of the model to optimize it
        optimizer = keras.optimizers.Adam(clipvalue=1.0)  # or use clipnorm=1.0
        model.compile(optimizer=optimizer, loss="mean_squared_error")
        # History objects provides details about training data
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True)

        history = model.fit(x_train, y_train, epochs=10, batch_size=32,
                            validation_split=0.2, callbacks=[early_stop], verbose=1)

        # Prepping values
        test_data = scaled_data[training - 500:, :]

        # Use the last 60 days of test data
        last_60_days = test_data.copy()

        # New predictions
        future_predictions = []

        for i in range(300):  # Predict for 186 days (approx. 6 months)
            print(f"Step {i+1}: last_60_days shape = {last_60_days.shape}")

            # Reshape last 60 days for prediction
            try:
                input_data = np.reshape(
                    last_60_days, (1, last_60_days.shape[0], 1))
            except Exception as e:
                print(f"Error during reshape: {e}")
                print(f"last_60_days: {last_60_days}")
                raise

            print(f"Step {i+1}: input_data shape = {input_data.shape}")

            # Predict the next day
            new_pred = model.predict(input_data)

            # Store the new prediction
            future_predictions.append(new_pred[0, 0])

            # Update last_60_days to include the latest prediction
            new_pred_reshaped = new_pred.reshape(
                1, 1)  # Ensure it's 2D for stacking
            last_60_days = np.vstack([last_60_days[1:], new_pred_reshaped])

        # Inverse transform to get the original scale
        future_predictions = np.array(future_predictions).reshape(-1, 1)
        future_predictions = scaler.inverse_transform(future_predictions)

        if ((future_predictions[len(future_predictions) - 1]) > (df.iloc[-1]["close"])):
            text = "This is a good stock"
        else:
            text = "This is a bad stock"

        plt.figure(figsize=(16, 8))
        plt.plot(np.arange(len(dataset)), dataset, label='Historical Data')
        plt.plot(np.arange(len(dataset), len(dataset) + len(future_predictions)),
                 future_predictions, label='Future Predictions', color='red')
        plt.xlabel('Days')
        plt.ylabel('Close Price')
        plt.title('Stock Price Predictions')
        plt.legend()

        img = io.BytesIO()

        plt.savefig(img, format="png")
        plt.close()
        img.seek(0)

        plot_url = base64.b64encode(img.getvalue()).decode('utf8')

    return render_template("answer.html", result=text, plot_url=plot_url)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)
