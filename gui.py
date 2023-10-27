import tkinter as tk
from tkinter import ttk
import pickle
import string
from nltk.corpus import stopwords
import nltk
from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)


def predict_spam(message):
    # 1. preprocess
    transformed_message = transform_text(message)
    # 2. vectorize
    vector_input = tfidf.transform([transformed_message])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. return the predicted spam label
    return result


# Load the vectorizer and model objects
tfidf = pickle.load(open("vectorizer.pkl", "rb"))
model = pickle.load(open('model.pkl', 'rb'))

# Create the tkinter window
root = tk.Tk()
root.title("Email/SMS Spam Classifier")

# Create the text input widget
message_input = ttk.Entry(root)
message_input.pack()

# Create the button widget
predict_button = ttk.Button(root, text="Predict", command=lambda: predict_spam_label())
predict_button.pack()

# Create the spam label widget
spam_label = ttk.Label(root, text="")
spam_label.pack()

# Function to predict the spam label and display it in the tkinter window
def predict_spam_label():
    message = message_input.get()
    result = predict_spam(message)

    if result == 1:
        spam_label.config(text="Spam")
    else:
        spam_label.config(text="Not Spam")

# Start the tkinter mainloop
root.mainloop()