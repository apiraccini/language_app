# dependencies
import pickle
import string
from pathlib import Path

# load trained models
BASE_DIR = Path(__file__).resolve(strict=True).parent
with open(f'{BASE_DIR}/ld_model.pkl', 'rb') as f:
    ld_model = pickle.load(f)
with open(f'{BASE_DIR}/oe_model.pkl', 'rb') as f:
    oe_model = pickle.load(f)

# define classes for prediction
classes1 = [
    'Arabic', 'Danish', 'Dutch', 'English', 'French', 'German',
    'Greek', 'Hindi', 'Italian', 'Kannada', 'Malayalam', 'Portugeese',
    'Russian', 'Spanish', 'Sweedish', 'Tamil', 'Turkish']
classes2 = ['Offensive', 'Not offensive']

# helper function
def preprocess(text, rm_emoji=False):
    
    text = text.replace('@USER', '') # remove mentions (@USER)
    text = text.replace('URL', '') # remove URLs
    text = text.replace('&amp', 'and') # replace ampersand (&) with and
    text = text.replace('&lt','') # remove &lt
    text = text.replace('&gt','') # remove &gt
    text = text.replace('\d+','') # remove numbers
    text = text.lower() # lowercase

    # remove punctuation
    for p in string.punctuation:
        text = text.replace(p, '')

    # remove emojis
    if rm_emoji:
        text = text.encode('ascii', 'ignore').decode('ascii')
    
    text = text.strip() # leading and trailing whitespaces

    return text

# prediction function for language detection model
def predict_ld(text):

    text = preprocess(text)
    pred = ld_model.predict([text])[0]
    prob = max(ld_model.predict_proba([text])[0])
    out = (classes1[pred], prob)

    return out

# prediction for offensive english classification model
def predict_oe(text):
    
    text = preprocess(text)
    pred = oe_model.predict([text])[0]
    prob = max(oe_model.predict_proba([text])[0])
    out = (classes2[pred], prob)

    return out