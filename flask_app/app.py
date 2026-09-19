from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


# Classe du modèle BiLSTM


class BiLSTMModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, output_dim=3, num_layers=2, dropout=0.6):
        super(BiLSTMModel, self).__init__()
        self.bilstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                              batch_first=True, bidirectional=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

    def forward(self, x):
        lstm_out, _ = self.bilstm(x)
        out = self.dropout(lstm_out[:, -1, :])
        out = self.fc(out)
        return out


# Configuration du device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Charger tokenizer et modèle BERT
tokenizer = AutoTokenizer.from_pretrained("aubmindlab/bert-base-arabertv02")
bert_model = AutoModel.from_pretrained(
    "aubmindlab/bert-base-arabertv02").to(device)
bert_model.eval()

# Fonction de chargement du modèle BiLSTM


def load_bilstm_model(model_path, device):
    if os.path.exists(model_path):
        model = BiLSTMModel(
            input_dim=768,
            hidden_dim=256,
            output_dim=3,
            num_layers=2,
            dropout=0.6
        ).to(device)

        state_dict = torch.load(model_path, map_location=device)

        model.load_state_dict(state_dict, strict=False)
        model.eval()
        return model
    else:
        raise FileNotFoundError(
            f"Le modèle à l'emplacement {model_path} n'a pas été trouvé.")


# Charger le modèle BiLSTM
try:
    model = load_bilstm_model('bilstm_model.pth', device)
    print("Modèle BiLSTM chargé avec succès!")
except Exception as e:
    print(f"Erreur de chargement: {str(e)}")
    model = None  # Si le modèle échoue à charger, mettre model à None

classes = ['negatif', 'neutre', 'positif']

# Fonction de prétraitement du texte


def preprocess_text(text):
    encoding = tokenizer(
        [text],
        truncation=True,
        padding='max_length',
        max_length=128,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
        outputs = bert_model(input_ids=input_ids,
                             attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]

    return cls_embedding.unsqueeze(1)  # [1, 1, 768]

# Fonction de prédiction du sentiment


def predict_sentiment(text):
    if model is None:
        raise RuntimeError("Le modèle BiLSTM n'a pas été chargé correctement.")
    inputs = preprocess_text(text)
    with torch.no_grad():
        outputs = model(inputs)
        probs = F.softmax(outputs, dim=1)
        predicted_idx = torch.argmax(probs, dim=1).item()
    return classes[predicted_idx]

# Route pour la page d'accueil


@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    error_message = None
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            try:
                prediction = predict_sentiment(text)
            except Exception as e:
                error_message = f"Erreur de prédiction: {str(e)}"
        else:
            error_message = "Le texte soumis est vide."
    return render_template('index.html', prediction=prediction, error_message=error_message)

# Route pour la page "À propos"


@app.route('/about')
def about():
    return render_template('about.html')

# Route pour la page "Contact"


@app.route('/contact')
def contact():
    return render_template('contact.html')

# Route pour la page d'analyse


@app.route('/analysis', methods=['GET', 'POST'])
def analysis():
    prediction = None
    error_message = None
    if request.method == 'POST':
        text = request.form.get('text')
        if text:
            try:
                prediction = predict_sentiment(text)
            except Exception as e:
                error_message = f"Erreur de prédiction: {str(e)}"
        else:
            error_message = "Le texte soumis est vide."
    return render_template('analysis.html', prediction=prediction, error_message=error_message)



@app.route('/api/predict', methods=['POST'])
def api_predict():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'No text provided'}), 400

    text = data['text']
    try:
        prediction = predict_sentiment(text)
        return jsonify({
            'prediction': prediction,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500
#


if __name__ == '__main__':
    app.run(port=5000, debug=True, host='0.0.0.0')