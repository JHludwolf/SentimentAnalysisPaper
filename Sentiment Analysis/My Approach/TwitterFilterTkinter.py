from tkinter import *
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

root = Tk()
root.geometry("300x300")
root.title(" Sentiment Analysis ")

LR_model = LogisticRegression(max_iter=150)
vec = TfidfVectorizer(min_df=5, max_df=0.8, sublinear_tf=True, use_idf=True)

def remove_punctuation(text):
    text = str(text)
    final = "".join(u for u in text if u not in ("?", ".", ";", ":", "!",'"', "'", "`"))
    return final

def to_lower(text):
    return text.lower()

def init():
    df = pd.read_csv('Datasets/Twitter Tweets/Tweets.csv')
    df['selected_text'] = df['selected_text'].apply(remove_punctuation)
    df['selected_text'] = df['selected_text'].apply(to_lower)
    X, Y = df['selected_text'], df['sentiment']
    trainX, _, trainY, _ = train_test_split(X, Y, test_size=0.2, random_state=0)
    train_vectors = vec.fit_transform(trainX.values.astype('U'))

    LR_model.fit(train_vectors, trainY)

def predict(text):
    return LR_model.predict(vec.transform([text]))[0].upper()

def Take_input():
    text = inputtxt.get("1.0", "end-1c")
    pred = predict(text.lower())

    Output.config(state=NORMAL)
    Output.delete('1.0', END)

    Output.insert(END, pred)
    Output.tag_add("here", "1.0", "2.0")
    Output.tag_config("here", justify='center')

    if(pred == 'POSITIVE'): Output.configure(bg='light green')
    elif(pred == 'NEUTRAL'): Output.configure(bg='light cyan')
    elif(pred == 'NEGATIVE'): Output.configure(bg='#cf7070')

    Output.config(state=DISABLED)
	
titleLabel = Label(text = "Enter a sentence:")

inputtxt = Text(root, height = 10, width = 25, bg = "light yellow")

Output = Text(root, height = 2, width = 25, bg = "light cyan")

Display = Button(root, height = 2, width = 20, text ="Predict", command = lambda:Take_input())

sentimentLabel = Label(text = "Sentiment Predicction:")

init()

titleLabel.pack()
inputtxt.pack()
sentimentLabel.pack()
Output.pack()
Display.pack()

mainloop()
