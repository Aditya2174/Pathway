# Setting up the env for the project

Make sure you have poetry installed. If not, install it using the following command
```bash
pip install poetry
```

Next install everything except pathway using poetry with the following command
```bash
poetry install --without pathway
```

Then install the local version of pathway using 
```bash
pip install -e PathwayPS
```

# Running the project
First run the server using the following command from the root directory
```bash
python src/server.py
```

Then run the client using the following command in a separate terminal
```bash
python src/pipeline.py
```
This command will run a streamlit interface that will allow you to interact with our solution.

# Setting up the data
The data is stored in the data folder. You can add the files into the data folder and the pipeline will automatically pick them up. Alternatively you can also 