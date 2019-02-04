import pandas as pd
from nltk.corpus import stopwords
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier

# Data Loading
data = pd.read_csv(r"turkish_cyberbullying.csv", encoding='utf-8')

# Dataset Columns
cyberbullying = data.cyberbullying
message = data.message

# Data Pre-processing
clean_messages_list = []
for message in data.message:
    message = message.lower()
    turkish_letters = '[^a-zçğışöü]'
    message = re.sub(turkish_letters, ' ', message)
    message = message.split()
    turkish_stop_words = stopwords.words('turkish')
    message = [word for word in message if not word in turkish_stop_words]
    message = " ".join(message)
    clean_messages_list.append(message)

# Data Splitting to Train and Test
x_train, x_test, y_train, y_test = train_test_split(
    clean_messages_list, cyberbullying, test_size=0.15, random_state=42)

# Bag of Words
count_vectorizer = CountVectorizer(max_features=1500)

# Feature Extraction
train_data_features = count_vectorizer.fit_transform(x_train).toarray()

# Data Evaluation
test_data_features = count_vectorizer.transform(x_test).toarray()


# Data Training Process with Classifiers
naive_bayes = GaussianNB()
naive_bayes.fit(train_data_features, y_train)

decision_tree = DecisionTreeClassifier()
decision_tree.fit(train_data_features, y_train)

adaboost = AdaBoostClassifier(DecisionTreeClassifier(), algorithm="SAMME")
adaboost.fit(train_data_features, y_train)

random_forest = RandomForestClassifier()
random_forest.fit(train_data_features, y_train)

# Prediction
y_pred = decision_tree.predict(test_data_features)

# Classification Score
print("Accuracy with Naive Bayes Classifier: ", naive_bayes.score(test_data_features, y_test))
print("Accuracy with AdaBoost Classifier: ", adaboost.score(test_data_features, y_test))
print("Accuracy with Random Forest Classifier: ", random_forest.score(test_data_features, y_test))
print("Accuracy with Decision Tree Classifier: ", decision_tree.score(test_data_features, y_test))
