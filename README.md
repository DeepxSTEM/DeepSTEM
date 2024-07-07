# Standalone Simple RAG Example

This is a simple standalone end-to-end LLM project showing a minimal RAG application that is ready to run with local free small LLMs or big (free or paid) LLMs on the cloud. 
Note: this RAG application only works with PDF files that do not have texts in image format. For the simple demonstration purpose I did not go into scenario of texts in image format. A future update may look into including PDF file with image texts.

Acknowledgement: This RAG application is adapted from the pixegami RAG tutorial channel:
Link: https://github.com/pixegami/rag-tutorial-v2

### Setup Steps

1. Clone this project to your computer:

   ```console
   Nevigate to the folder where you want to place the project then issue the git clone command below
   git clone "https://github.com/DeepxSTEM/DeepSTEM.git"
   
   ```

2. Create a python virtual environment and activate it:

   For Windows: replace "env_name" with your desired environment name.

   ```console
   python -m venv env_name
   env_name\Scripts\activate
   ```

   For Mac: replace "env_name" with your desired environment name.
   
   ```console
   python3 -m venv myenv
   source myenv/bin/activate
   ```

3. Nevigate to the root repository, `DeepSTEM`, install the requirements:

   ```console
   pip install -r requirements.txt
   ```

4. Refer to [Get an API Key for the Mixtral 8x7B Instruct API Endpoint](https://nvidia.github.io/GenerativeAIExamples/latest/api-catalog.html#get-an-api-key-for-the-mixtral-8x7b-instruct-api-endpoint)
   for information about how to get an NVIDIA API key.

   ```.env file
   GOOGLE_API_KEY="your google api key"
   NVIDIA_API_KEY="your Nvidia api key"
   ```

5. Go to https://www.ollama.com/ and download Ollama to your computer or issue the command:
   pip install ollama

   Click 'Models' and find the model your computer can handle then issue the following command to download the model to your computer.
   ollama pull model_name (e.g., Ollama pull mistral)

6. If you want to use Google Gemini LLM, OpenAI LLM or Nvidia-mistralai/mixtral-8x7b-instruct-v0.1, 
   add your Google API key, OpenAI API key or Nvidia API key to the .env file, then uncomment those LLMs that you want to send your query to.

7. Run the example using Streamlit:

   ```console
   streamlit run Hybrid_RAG.py
   ```

8. Test the deployed example by going to `http://localhost:8501` in a web browser, if it does not 
   automatically launched.
   
   (A): Click the 'Browser files' button to select your PDF files then click the 'Upload' button
        to upload your PDF files.
   (B): Enter your question/query then hit Enter to start chating with the AI Assistant

You are all set now! Try out queries related to the knowledge base using text from the user interface.
