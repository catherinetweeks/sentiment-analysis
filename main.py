import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

#Step 1: Dataset Exploration
df = pd.read_csv('Tweets.csv')
df.head()
#This reads the data from the CSV file into a Pandas DataFrame.
#It then displays the first 5 rows of the DataFrame to give a preview of data.

#Step 2: Data Preprocessing
# Convert text to lowercase
df['text'] = df['text'].str.lower()
df['text'] = df['text'].astype(str) # Convert the 'text' column to string
df['tokens'] = df['text'].apply(nltk.word_tokenize)  # Tokenization
# Remove stopwords
stopwords = nltk.corpus.stopwords.words('english')
df['tokens'] = df['tokens'].apply(lambda x: [word for word in x if word not in stopwords])

#Step 3: Split the Data Set
X = df['text']
y = df['sentiment']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#Split data into training data and testing data
#20% of data for testing, and 80% will be used for training (test_size=0.2)
#Random_state=42 uses a seed number to insure that the split will be the same each time

#Step 4: Feature Extraction
vectorizer = TfidfVectorizer()
X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)
#Creating TF-IDF vectorizer, fitting it to training data and transforming it.

#Step 5: Build and train a sentiment analysis model
model = SVC()
print("Starting model training...")
model.fit(X_train_vectors, y_train)
print("Model training completed.")
#Support Vector Machine = machine learning algotithm. Default SVC() will use a linear kernel
#This method trains the SVC model on training data

#Step 6: Evaluate the model
y_pred = model.predict(X_test_vectors)
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Accuracy Score:", accuracy_score(y_test, y_pred))