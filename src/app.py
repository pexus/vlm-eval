# app.py - Main Flask Application
import json
import os
import requests
import configparser
from transformers import AutoModelForCausalLM, AutoProcessor, AutoModelForVision2Seq, AutoConfig, Qwen2VLForConditionalGeneration
from PIL import Image
import torch
import logging
import gc
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from io import BytesIO
# Update the import statements to include MllamaForConditionalGeneration
from transformers import MllamaForConditionalGeneration

# Set up logger
logging.basicConfig(level=logging.INFO)  # Set to INFO to turn off debug outputs
logger = logging.getLogger(__name__)

# Load API token from config.ini
config = configparser.ConfigParser()
config.read('config.ini')
api_token = config.get('huggingface', 'api_token', fallback=None)

# Set up Flask app
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'jpeg', 'jpg', 'png', 'gif'}

# Available VLM models
MODEL_CHOICES = {
    "google/paligemma-3b-mix-448": "float16",
    "Qwen/Qwen2-VL-2B-Instruct": "main",
    "microsoft/Phi-3.5-vision-instruct": "main"
}

# Load prompts from a file
PROMPT_FILE = 'prompts.json'

# Load prompts from the file if it exists, else use default prompts
if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE, 'r') as file:
        PROMPTS = json.load(file)
else:
    PROMPTS = [
        "Extract texts from the image.",
        "Extract texts from the image and return each text string in a new line.",
        "Extract texts from the image and return each text string in a new line. The extracted text should be clean and readable.",
    ]

@app.route('/add_prompt', methods=['POST'])
def add_prompt():
    new_prompt = request.form.get('new_prompt')
    if new_prompt and new_prompt not in PROMPTS:
        PROMPTS.append(new_prompt)
        # Save the updated prompts list to the file
        with open(PROMPT_FILE, 'w') as file:
            json.dump(PROMPTS, file)
    return jsonify({'status': 'success', 'prompts': PROMPTS})




def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_model_and_processor(model_name, revision):
    logger.info(f"Loading model and processor for {model_name} with revision {revision}...")
    if model_name == "Qwen/Qwen2-VL-2B-Instruct":
        # Load the specific model and processor for Qwen2-VL-2B Instruct
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct",
            torch_dtype="auto",
            device_map="auto",
            use_auth_token=api_token
        )
        processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    elif model_name == "microsoft/Phi-3.5-vision-instruct":
        # Load the specific model and processor for Phi-3.5-vision-instruct
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,  # Quantized version to reduce memory usage
            _attn_implementation='eager'
        )
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, num_crops=4)
    else:
        # Load other models using AutoModelForVision2Seq
        processor = AutoProcessor.from_pretrained(model_name, use_auth_token=api_token)
        config = AutoConfig.from_pretrained(model_name, use_auth_token=api_token, revision=revision)
        max_input_tokens = getattr(config, 'max_position_embeddings', 1024)
        logger.info(f"Maximum number of input tokens: {max_input_tokens}")
        model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            use_auth_token=api_token,
            revision=revision,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return model, processor

@app.route('/')
def index():
    return render_template('index.html', model_choices=MODEL_CHOICES.keys(), prompts=PROMPTS)

@app.route('/upload', methods=['POST'])
def upload_file():
    model_name = request.form.get('model')
    if model_name not in MODEL_CHOICES:
        return "Invalid model selected", 400
    revision = MODEL_CHOICES[model_name]

    # Load the selected model and processor if not already loaded
    global model, processor
    if 'model' not in globals() or model_name != getattr(model, 'name', None):
        # Free up CUDA memory and delete the previous model if it exists
        if 'model' in globals():
            del model
            torch.cuda.empty_cache()
            gc.collect()
        model, processor = load_model_and_processor(model_name, revision)
        model.name = model_name  # Store model name to prevent reloading the same model

    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Load the uploaded image
        image = Image.open(file_path)
        if image.format == 'GIF':
            image = image.convert('RGBA')
        else:
            image = image.convert('RGB')

        # Extract text from the image
        prompt = request.form.get('prompt')
        text_segments = extract_text_from_image(image, model_name, prompt)

        # Render the result page with the image and extracted text
        return render_template('result.html', model_name=model_name, prompt=prompt, image_url=file_path, text_segments=text_segments)
    else:
        return "Unsupported file type", 400


@app.route('/download_prompts', methods=['GET'])
def download_prompts():
    selected_prompt = request.args.get('selected_prompt')
    if selected_prompt:
        prompts_to_download = [selected_prompt]
    else:
        prompts_to_download = PROMPTS

    response = jsonify(prompts=prompts_to_download)
    response.headers['Content-Disposition'] = 'attachment; filename=prompts.json'
    return response

@app.route('/extract_text', methods=['POST'])
def extract_text_from_image(image, model_name, prompt):
    # Load the selected model and processor if not already loaded
    if model_name == "Qwen/Qwen2-VL-2B-Instruct":
        # Properly format the prompt for Qwen2-VL-2B Instruct model
        messages = [
            {"role": "user", "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]}
        ]
        formatted_prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = processor(
            image,
            formatted_prompt,
            add_special_tokens=False,
            return_tensors='pt'
        ).to(model.device)
    elif model_name == "google/paligemma-3b-mix-448":
        # Handle text extraction for Paligemma model
        image = image.resize((448, 448))  # Resizing to a common dimension that the model can handle
        inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
        inputs = {key: value.to('cuda') if torch.cuda.is_available() else value for key, value in inputs.items()}
    elif model_name == "microsoft/Phi-3.5-vision-instruct":
        # Handle text extraction  Phi-3.5-vision-instruct model
        formatted_prompt = f"<|image_1|>\n{prompt}<|end|>\n<|assistant|>\n"
        inputs = processor(formatted_prompt, [image], return_tensors='pt').to(model.device)
    
    # Generate output from the image
    logger.info("Generating output from the model...")
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,  # Reduce to limit response length
        eos_token_id=processor.tokenizer.eos_token_id,  # Explicitly set end-of-sequence token
        do_sample=False  # Disable sampling to make the output more deterministic
    )

    # Remove input tokens if necessary
    if model_name in ["microsoft/Phi-3.5-vision-instruct", "Qwen/Qwen2-VL-2B-Instruct"]:
        output_ids = output_ids[:, inputs['input_ids'].shape[1]:]

    # Decode the output
    generated_text = processor.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

      
    # Clean and process the generated text
    generated_text_clean = generated_text[0].strip()
    text_segments = [line.strip() for line in generated_text_clean.split('\n') if line.strip()]

    

    # Free up CUDA memory
    del inputs
    del output_ids
    torch.cuda.empty_cache()  # Clear memory for the default CUDA device

    # If there are multiple GPUs, free the cache for each one
    if torch.cuda.device_count() > 1:
        for device_id in range(torch.cuda.device_count()):
            with torch.cuda.device(device_id):
                torch.cuda.empty_cache()

    gc.collect()  # Python garbage collection
    return text_segments

# Route to serve uploaded files
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)


    