# Standalone Simple RAG Example

This is a simple standalone implementation showing a minimal RAG application that can run with local free small LLMs or big free or paid LLMs on the cloud. 

### Setup Steps

1. Clone this project to your computer:

   ```comsole
   Nevigate to the folder where you want to place the project then issue the git clone command below
   git clone "https://github.com/DeepxSTEM/DeepSTEM.git"
   
   ```

2. Create a python virtual environment and activate it:

   ```comsole - Replace "env_name" with your desired environment name.
   python -m venv env_name
   env_name\Scripts\activate
   ```

3. Nevigate to the root repository, `DeepSTEM`, install the requirements:

   ```console
   pip install -r requirements.txt
   ```

4. If you want to use Google LLM or mistralai/mixtral-8x7b-instruct-v0.1, add your Google API key or 
   Nvidia API key to the .env file:

   Refer to [Get an API Key for the Mixtral 8x7B Instruct API Endpoint](https://nvidia.github.io/GenerativeAIExamples/latest/api-catalog.html#get-an-api-key-for-the-mixtral-8x7b-instruct-api-endpoint)
   for information about how to get an NVIDIA API key.

   ```.env file
   GOOGLE_API_KEY="your google api key"
   NVIDIA_API_KEY="your Nvidia api key"
   ```

5. Go to https://www.ollama.com/ and download Ollama to your computer or issue the command:
   pip install ollama

   Click 'Models' and find the model your computer can handle then issue the following command to download the model to your computer.
   ollama pull model_name (e.g., Ollama pull mistral)

6. Run the example using Streamlit:

   ```console
   streamlit run Hybrid_RAG.py
   ```

7. Test the deployed example by going to `http://localhost:8501` in a web browser.
   Enter your question then hit Enter to test the model

You are all set now! Try out queries related to the knowledge base using text from the user interface.
