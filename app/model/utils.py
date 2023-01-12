# dependencies
import pickle
import re
from pathlib import Path

# load trained model
BASE_DIR = Path(__file__).resolve(strict=True).parent
with open(f'{BASE_DIR}/model_110123.pkl', 'rb') as f:
    model = pickle.load(f)

# define classes for prediction
classes = [
    'Arabic', 'Danish', 'Dutch', 'English', 'French', 'German',
    'Greek', 'Hindi', 'Italian', 'Kannada', 'Malayalam', 'Portugeese',
    'Russian', 'Spanish', 'Sweedish', 'Tamil', 'Turkish']

# prediction function
def predict_model(text):
    '''Predict the text language using the trained pipeline'''
    
    text = re.sub(r'[!@#$(),\n"%^*?\:;~`0-9]', " ", text)
    text = text.lower()
    pred = model.predict([text])
    
    return classes[pred[0]]