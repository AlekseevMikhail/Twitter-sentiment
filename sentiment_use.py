from keras.preprocessing.sequence import pad_sequences
import pickle
from keras.models import load_model

# Load model
model = load_model('best_model.h5')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

def predict_class(text):
    sentiment_classes = ['Negative', 'Neutral', 'Positive']
    max_len = 50

    # Transforms text to a sequence of integers using a tokenizer object
    xt = tokenizer.texts_to_sequences(text)
    # Pad sequences to the same length
    xt = pad_sequences(xt, padding='post', maxlen=max_len)
    # Do the prediction using the loaded model
    yt = model.predict(xt).argmax(axis=1)
    # Print the predicted sentiment
    print('The predicted sentiment is', sentiment_classes[yt[0]])


predict_class(['"I hate when I have to call and wake people up'])

predict_class(['The food was meh'])

predict_class(['He is a best minister india ever had seen'])