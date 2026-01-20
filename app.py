from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences  # Fixed Import
import numpy as np
import pickle

# ---  DEFINE CUSTOM LAYER ---
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super(BahdanauAttention, self).__init__()
        self.W1 = tf.keras.layers.Dense(units)
        self.W2 = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, query, values):
        query_with_time_axis = tf.expand_dims(query, 1)
        score = self.V(tf.nn.tanh(self.W1(query_with_time_axis) + self.W2(values)))
        attention_weights = tf.nn.softmax(score, axis=1)
        context_vector = attention_weights * values
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, attention_weights

# ---  INITIALIZE FLASK APP ---
app = Flask(__name__)  


# Load Tokenizer (The Dictionary)
with open('Models/tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load Models
encoder_model = load_model('Models/encoder_model.keras')
decoder_model = load_model('Models/decoder_model.keras', custom_objects={'BahdanauAttention': BahdanauAttention})

reverse_word_index = tokenizer.index_word
print("Models Loaded!")

# --- INFERENCE FUNCTIONS ---
def decode_sequence(input_seq):
    enc_out, state_h, state_c = encoder_model.predict(input_seq, verbose=0)
    target_seq = np.zeros((1, 1))
    target_seq[0, 0] = tokenizer.word_index.get('start', 1) 

    stop_condition = False
    decoded_sentence = ""

    while not stop_condition:
        output_tokens, h, c = decoder_model.predict(
            [target_seq, enc_out, state_h, state_c], 
            verbose=0)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_word_index.get(sampled_token_index, '')
        if (sampled_char == 'end' or len(decoded_sentence.split()) > 20):
            stop_condition = True
        else:
            decoded_sentence += ' ' + sampled_char
        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = sampled_token_index
        state_h, state_c = h, c

    return decoded_sentence.strip()

def predict_headline(text):
    # Clean text
    text = text.lower()
    seq = tokenizer.texts_to_sequences([text])
    pad_seq = pad_sequences(seq, maxlen=60, padding='post')
    
    result = decode_sequence(pad_seq)
    return result

# ---  ROUTES ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def generate_headline():
    user_input = request.form['article_text']
    summary = predict_headline(user_input)
    return render_template('index.html', prediction=summary)

if __name__ == '__main__':
    app.run(debug=True)