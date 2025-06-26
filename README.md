# tweet-sentiment-classifier
This project applies natural language processing and machine learning to classify public tweets directed at U.S. airlines by sentiment—negative, neutral, or positive. The objective is to support a customer service team by flagging negative tweets for human attention and automatically acknowledging positive ones.


## Project Context

Airlines receive a high volume of customer feedback via Twitter. As a mock data scientist on the social team, the task was to build a sentiment classifier that can help triage incoming tweets. The model was trained on a labeled dataset of historical tweets, with sentiment labels manually assigned.

## Data

The training and test data are modified subsets of the Twitter US Airline Sentiment dataset from Kaggle. The available columns include:

- `text`: Full content of the tweet
- `airline_sentiment`: Sentiment label (negative, neutral, or positive)
- `airline`: Airline referenced in the tweet (not used in the model)

Data is provided in Parquet format:
- `tweets-train.parquet`
- `tweets-test.parquet`

The data was loaded and prepared using Pandas. Only the tweet text was used to train models.

## Methods

### Preprocessing

Tweets were vectorized using `CountVectorizer` and `TfidfVectorizer` to transform text data into numerical format. A pipeline was created to handle the transformation and modeling steps in sequence.

### Modeling

A machine learning pipeline was constructed using scikit-learn. Several models were tested, including:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest Classifier

Model selection and hyperparameter tuning were done using `GridSearchCV` with cross-validation. The pipeline outputs the best-performing model.

### Evaluation

The final model was evaluated on a held-out test set using standard classification metrics:

- Accuracy
- Precision, Recall, F1 Score (macro-averaged)

Model performance was tracked separately for each sentiment class to assess how well neutral and minority classes were predicted.

## Implementation Notes

- Preprocessing and modeling were implemented using a `Pipeline` object
- Text feature extraction used a maximum of 100 most common words
- The model includes `fit`, `predict`, and `predict_proba` methods as required
- The model is compatible with scikit-learn ≥1.6.1 and serialized using `joblib` ≤5MB
- Cross-validation was used to avoid overfitting and select hyperparameters


