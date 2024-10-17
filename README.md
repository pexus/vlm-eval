A simple flask web app to evaluate small VLMs (Vision Language Models). Currently it includes the following models from the Hugging face. 
- google/paligemma-3b-mix-448
- Qwen/Qwen2-VL-2B-Instruct
- microsoft/Phi-3.5-vision-instruct

The application with VLM running loaded and inferencing locally was tested on a consumer grade desktop with following configuration.

_Overall the base Qwen/Qwen2-VL-2B-Instruct model performed the best with very good accuracy for extracting text from images._

## System Information

```plaintext
System:
  Host: ai-desktop Kernel: 6.8.0-45-generic arch: x86_64 bits: 64
  Desktop: GNOME v: 46.0 Distro: Ubuntu 24.04 LTS (Noble Numbat)
  32 GB Memory
Machine:
  Type: Desktop Mobo: ASUSTeK model: M5A97 LE R2.0 v: Rev 1.xx
    serial: <superuser required> UEFI: American Megatrends v: 2601
    date: 03/24/2015
CPU:
  Info: 8-core model: AMD FX-8350 bits: 64 type: MT MCP cache: L2: 8 MiB
  Speed (MHz): avg: 1489 min/max: 1400/4000 cores: 1: 1407 2: 2100 3: 1400
    4: 1400 5: 1400 6: 1404 7: 1405 8: 1400
Graphics:
  Device-1: NVIDIA GA106 [GeForce RTX 3060 Lite Hash Rate] driver: nvidia
    v: 535.183.01  12 GB GPU Memory
  Device-2: NVIDIA GA106 [GeForce RTX 3060 Lite Hash Rate] driver: nvidia
    v: 535.183.01  12 GB GPU Memory
```

## Clone the Repository

To clone this repository to your local machine, run the following command:

```bash
git clone https://github.com/pexus/vlm-eval.git
```

## Installation

To install all the dependencies for this project, run the following command:

```bash
pip install -r requirements.txt

```
## Setting up Hugging Face API Key

To download the models for this project, you will need an API key from Hugging Face. Follow the steps below:

1. **Create an Account on Hugging Face**:
   - Visit [Hugging Face](https://huggingface.co/) and sign up for a free account if you don't already have one.

2. **Create an API Key**:
   - After logging in, navigate to your account settings and create a new API token.

3. **Create a `config.ini` File**:
   - In the `src` folder of this project, create a file named `config.ini`.
   - Add your Hugging Face API token in the following format:

   ```ini
   [huggingface]
   api_token = YOUR_HUGGINGFACE_API_TOKEN

## Running the Application

To run the application, make sure you have all the dependencies installed (as listed in the `requirements.txt`), and then execute the following command:

```bash
python app.py
```
Once the application starts, you will see the URL where it can be accessed in your terminal. Typically, Flask applications run on http://127.0.0.1:5000 by default unless configured otherwise.


* Serving Flask app 'app'
* Debug mode: off
WARNING: This is a development server. Do not use it in a production deployment.
Use a production WSGI server instead.
* Running on http://127.0.0.1:5000 (Press CTRL+C to quit)

After the application starts, open your browser and go to the URL shown in the output (usually http://127.0.0.1:5000) to access the application.



