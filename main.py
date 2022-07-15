# https://www.youtube.com/watch?v=-ykeT6kk4bk&t=1240s
# https://dirty-text-classification.herokuapp.com/docs#  link to the url endpoint
import pandas as pd
from river.naive_bayes import MultinomialNB
from river.feature_extraction import BagOfWords, TFIDF
from river.compose import Pipeline
from fastapi import FastAPI

data = pd.read_csv('fanfic_test.csv')
data = list(zip(data.story, data.rating))
# model building
model = Pipeline(('vectorizer', BagOfWords(lowercase=True)), ('nv', MultinomialNB()))
for text, label in data:
    model = model.learn_one(text, label)
app = FastAPI()

@app.get("/")
def about():
    return {"test": "Me"}

@app.get("/about")
def about():
    return {"About": "Me"}



@app.get("/text-classification/{fanfic_text}")
def get_text_classification(fanfic_text: str, get_word: bool = False):
    base_dic = {"text": fanfic_text, "prediction": model.predict_one(fanfic_text),
                'probability': model.predict_proba_one(fanfic_text)}
    if get_word == False:
        return base_dic
    else:
        message_list = fanfic_text.split()  
        predict_dic = {'word': [], 'prediction': [], 'explicit_prob': [], 'general_prob': []}
        for index, word in enumerate(message_list):
            remove_list = message_list.copy()
            removed_word = remove_list.pop(index)
            new_word = " ".join(remove_list)
            new_pred = model.predict_one(new_word)
            prob_dic = model.predict_proba_one(new_word)
            predict_dic['word'].append(removed_word)
            predict_dic['prediction'].append(new_pred)
            predict_dic['explicit_prob'].append(prob_dic['explicit'])
            predict_dic['general_prob'].append(prob_dic['general'])
        base_dic.update( {'word_prediction' : predict_dic} )
        return base_dic


if __name__ == "__main__":
    app.run()