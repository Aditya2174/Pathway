API Tokens
-
Setting up the tokens for the repository
Required tokens:
- HF_TOKEN - Required for gated models (Llama Guard 3 8B)
- GOOGLE_API_KEY - Required for the gemini model
- TAVILY_API_KEY - Reqquired for the Tavily Search Engine

Creating HF_TOKEN:
-
Go to the [model page](https://huggingface.co/meta-llama/Llama-Guard-3-8B) and fill the access form to get the access. You will get access in about half an hour. Meanwhile generate your hugging face token by signing in to [HuggingFace](https://huggingface.co/). Go to Profile > Access Tokens > Create New Token. Give the token any name and click create token.

Creating GOOGLE_API_KEY:
-
Go to Gemini create api key [page](https://aistudio.google.com/app/apikey) and follow the instructions to generate the api key.

Creating TAVILY_API_KEY:
-
Go to [Tavily](https://tavily.com/) and click on get started and sign in. There will be an option to create api token on the landing page.

<b>Note: Keep all these tokens in a .env file.<b>

Docker
-
For running the code through docker, you can directly run ```docker compose up```

Local
-
If you want to run without docker, follow the given steps:
- Setup the dependencies through ```poetry install``` after installing [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer)
- Add other dependencies like [torch](https://pytorch.org/get-started/locally/)
- Install the local Pathway through ```pip install -e 'PathwayPS[all]'```. This will require rust and cmake to be installed on system. Any other additional dependencies can be figured out from the given DockerFile
- Change the PATHWAY_HOST to 127.0.0.1 in src/server.py and change the PATHWAY_HOST in src/frontend.py to the same value.
- Run the server using `python src/server.py` to run the server 
- Run the client using `streamlit run src/frontend.py` 

<B> Note: Building the server can take time depending on the number of files in the data folder. Run the frontend only after server is done building

Compute Requirements
-
The main compute requirements arise from 2 things

- Models
--

The pipeline itself is built using the Gemini api and light weight models. There is only one big LLM which we are using for guardrails since it is a task that should be done using big models. This brings our GPU pesk usage to around 17-18 GB. For running the evaluations we used 1 Nvidia RTX A5000 GPU.

- RAM
--

Embedding large PDFs require a large amount of RAM. In case the RAM is insufficient the server will be eventually terminated. Our evaluations were ran on a server with around 350 GB of RAM and 20 CPU cores.