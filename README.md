# NYC Pothole Repair Prediction App

This Streamlit app predicts the number of days required to fix a pothole in NYC. The prediction model was trained in a different project and has been pickled for use in this application. The app leverages the model, dataset, and visualizations to provide insights and predictions.

## Features

- **Prediction Model**: Uses a pre-trained and pickled machine learning model to predict the number of days required to fix a pothole.
- **Dataset**: Utilizes a dataset of pothole repairs in NYC.
- **Visualizations**: Displays various visualizations related to pothole repairs in NYC.

## Getting Started

These instructions will help you set up and run the project on your local machine for development and testing purposes.

### Prerequisites

Ensure you have the following installed on your local development machine:

- [Python 3.7+](https://www.python.org/downloads/)
- [Streamlit](https://streamlit.io/)

### Installation

1. **Clone the repository:**

    ```bash
    git clone https://github.com/yourusername/nyc-pothole-prediction.git
    ```

2. **Navigate to the project directory:**

    ```bash
    cd nyc-pothole-prediction
    ```

3. **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

### Running the App

To run the Streamlit app, use the following command:

```bash
streamlit run app.py
