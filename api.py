from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import json
import os
import math
from transformers import GPT2Tokenizer

app = Flask(__name__)
CORS(app, resources={
    r"/chat": {"origins": "*"},
    r"/health": {"origins": "*"}
})

model = None
tokenizer = None
config = None

class ProfessionalConfig:
    def __init__(self):
        self.vocab_size = 50257
        self.max_length = 256
        self.d_model = 768
        self.n_layers = 12
        self.n_heads = 12
        self.d_ff = 3072
        self.dropout = 0.1
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.temperature = 0.7
        self.top_k = 50
        self.top_p = 0.9
        self.repetition_penalty = 1.2
        self.beam_size = 3
        self.context_window = 3

def load_model():
    global model, tokenizer, config
    
    try:
        config = ProfessionalConfig()
        
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.add_special_tokens({
            'additional_special_tokens': [
                '<sep>', '<res>', '<saf>', '<usr>', 
                '<sys>', '<emo>', '<therapist>'
            ]
        })
        
        model = EnhancedTransformer(config, tokenizer)
        
        if os.path.exists('best_model_weights.pth'):
            model.load_state_dict(torch.load('best_model_weights.pth', map_location=config.device))
        
        model.eval()
        print("Model loaded successfully!")
        
    except Exception as e:
        print(f"Error loading model: {str(e)}")

class EnhancedTransformer(torch.nn.Module):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        
        from transformers import GPT2LMHeadModel, GPT2Config
        gpt2_config = GPT2Config.from_pretrained('gpt2')
        self.gpt2 = GPT2LMHeadModel.from_pretrained('gpt2', config=gpt2_config)
        self.gpt2.resize_token_embeddings(len(tokenizer))
        self.gpt2.config.pad_token_id = tokenizer.eos_token_id
        
        self.to(config.device)
    
    def generate(self, prompt, max_length=100, context=None, **kwargs):
        self.eval()
        
        crisis_response = self._detect_crisis(prompt)
        if crisis_response:
            return crisis_response
        
        context_ids = []
        if context:
            for c in context[-self.config.context_window:]:
                context_ids.extend(self.tokenizer.encode(c) + [self.tokenizer.sep_token_id])
        
        input_ids = context_ids + self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], device=self.config.device)
        
        output = self.gpt2.generate(
            input_ids=input_tensor,
            max_length=max_length + len(input_ids),
            temperature=self.config.temperature,
            top_k=self.config.top_k,
            top_p=self.config.top_p,
            repetition_penalty=self.config.repetition_penalty,
            num_beams=self.config.beam_size,
            early_stopping=True,
            pad_token_id=self.tokenizer.eos_token_id,
            no_repeat_ngram_size=3,
            do_sample=True
        )
        
        response = self.tokenizer.decode(output[0][len(input_ids):])
        return self._post_process_response(prompt, response)
    
    def _detect_crisis(self, text):
        crisis_keywords = {
            'suicide': ["kill myself", "end it all", "suicide", "want to die"],
            'selfharm': ["cutting", "self harm", "hurt myself"],
            'abuse': ["being abused", "domestic violence", "rape"]
        }
        
        resources = {
            'US': {
                'suicide': "988 Suicide & Crisis Lifeline: Call or text 988",
                'selfharm': "Crisis Text Line: Text HOME to 741741",
                'abuse': "National Domestic Violence Hotline: 1-800-799-7233"
            }
        }
        
        text_lower = text.lower()
        for category, keywords in crisis_keywords.items():
            if any(kw in text_lower for kw in keywords):
                return f"<saf>I'm very concerned about what you're sharing. Please contact {resources['US'][category]} immediately. You're not alone.</saf>"
        return None
    
    def _post_process_response(self, prompt, response):
        response = response.split('\n')[0]  
        response = response.rsplit('.', 1)[0] + '.' if '.' in response else response
        
        if not any(p in response for p in ["How", "What", "Tell me", "understand"]):
            if not response.startswith(("I ", "You ")):
                response = "I understand. " + response
        
        if any(word in prompt.lower() for word in ['therapy', 'counseling', 'professional']):
            if 'resource' not in response.lower():
                response += " Would you like information about finding a therapist in your area?"
        
        return response

load_model()

DEFAULT_RESPONSES = {
    "hello": "Hello! How are you feeling today?",
    "hi": "Hi there! I'm Healio, your mental health companion.",
    "sad": "I'm sorry to hear that. Would you like to talk about what's bothering you?",
    "depressed": "I'm here for you. Have you considered speaking with a professional about how you're feeling?",
    "stress": "Stress can be tough. Try taking some deep breaths or going for a short walk.",
    "anxious": "Anxiety can be challenging. Try focusing on your breathing for a moment.",
    "default": "I'm here to listen. Tell me more about what's on your mind."
}

@app.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({'error': 'No message provided'}), 400
            
        user_message = data['message'].lower()
        
        for keyword, response in DEFAULT_RESPONSES.items():
            if keyword in user_message:
                return jsonify({
                    'response': response,
                    'status': 'success'
                })
        
        if model and tokenizer:
            response = model.generate(
                prompt=user_message,
                max_length=150
            )
            return jsonify({
                'response': response,
                'status': 'success'
            })
        
        return jsonify({
            'response': DEFAULT_RESPONSES['default'],
            'status': 'success'
        })
    
    except Exception as e:
        print(f"Error in chat endpoint: {str(e)}")
        return jsonify({
            'response': "I'm having some trouble understanding. Could you rephrase that?",
            'status': 'error'
        }), 500

@app.route('/health', methods=['GET', 'POST'])
def health_check():
    return jsonify({
        'status': 'running',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)