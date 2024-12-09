from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)

class ModelSingleton:
    """Singleton class to load and reuse the model and tokenizer."""
    _instance = None  # Class-level attribute to hold the singleton instance

    def __new__(cls, model_path, *args, **kwargs):
        if cls._instance is None:
            # First time initialization
            cls._instance = super(ModelSingleton, cls).__new__(cls)
            cls._instance._initialize(model_path, *args, **kwargs)
        return cls._instance

    def _initialize(self, model_path, device="cuda" if torch.cuda.is_available() else "cpu"):
        """Initialize the model and tokenizer."""
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()  # Set the model to evaluation mode
        self.tokenizer.pad_token = self.tokenizer.eos_token  # Optional if required by the model
    
    def generate_response(self, prompt, max_length=500):
        """Generate a response using the preloaded model."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(output[0], skip_special_tokens=True)

# Specify the model checkpoint path
best_checkpoint_path = "/mnt/c/Users/zeiad/Downloads/Llama implementaion/output/checkpoint-3000"

# Load the model once
model_instance = ModelSingleton(best_checkpoint_path)

@app.route("/generate", methods=["POST"])
def generate():
    """API endpoint to generate a response."""
    data = request.get_json()
    prompt = data.get("prompt", "")
    if not prompt:
        return jsonify({"error": "Prompt is required"}), 400

    # Generate response using the model
    response = model_instance.generate_response(prompt)
    return jsonify({"response": response})

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)