from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re
import os

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


# Utility function to extract data dynamically
def extract_data(input_text):
    try:
        # Extract the topic
        topic_match = re.search(r"Topic:\s*(.*)", input_text, re.IGNORECASE)
        topic = topic_match.group(1).strip() if topic_match else "No topic found"
        
        # Extract the question
        question_match = re.search(r"Question:\s*(.*?)(?=Choices:|Correct Answer:|$)", input_text, re.IGNORECASE | re.DOTALL)
        question = question_match.group(1).strip() if question_match else "No question found"
        
        # Extract the choices
        choices_match = re.search(r"Choices:\s*(.*?)(?=Correct Answer:|$)", input_text, re.IGNORECASE | re.DOTALL)
        choices_text = choices_match.group(1).strip() if choices_match else ""
        choices = re.findall(r"([A-D])\s+(.*?)(?=(?:,|$))", choices_text)
        choices_dict = {key: value.strip() for key, value in choices}
        
        # Extract the correct answer
        correct_answer_match = re.search(r"Correct Answer:\s*(.*)", input_text, re.IGNORECASE)
        correct_answer = correct_answer_match.group(1).strip() if correct_answer_match else "No correct answer found"
        
        return {
            "topic": topic,
            "question": question,
            "choices": choices_dict,
            "correct_answer": correct_answer
        }
    
    except Exception as e:
        raise ValueError(f"Error parsing text: {str(e)}")


# Specify the model checkpoint path
best_checkpoint_path = "/mnt/c/Users/zeiad/Downloads/Llama implementaion/output/checkpoint-3000"

# Load the model once
model_instance = ModelSingleton(best_checkpoint_path)

@app.route('/recieve', methods=['GET'])
def get_data():
    # Get the query parameters from the URL
    data = request.get_json()
    topic = data.get('topic')
    explanation = data.get('explanation')
    answer = data.get('correct_answer')

    # Validate that all required parameters are provided
    if not all([topic, explanation, answer]):
        return jsonify({"error": "Missing one or more required parameters: topic, explanation, answer."}), 400

    # Process the data (for demonstration, just return the received data)
    
    prompt =f"""<|im_start|> Based on the Topic: {topic} and the Explanation: {explanation}. Correct Answer: {answer}
                Generate a question related to the explanation containing age and gender of patient as well as 4 multiple 
                choice options without duplicating them and differet relevant diseases. The options should include the correct answer 
                and generate the other 3 based on the standard terminology.<|im_end|>"""

    
    generated_response = model_instance.generate_response(prompt)

    return generated_response, 200

@app.route("/generated_response", methods=["POST"])
def generate():

    # Generate response using the model
    generated_response = get_data()
    MCQ_Object = extract_data(generated_response)

    return MCQ_Object

if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "false").lower() == "true"
    app.run(host="0.0.0.0", port=8080, debug=debug_mode)