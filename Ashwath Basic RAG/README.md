There are 3 files: server_runner.py, langchain_frontend.py and testing.py

Run the server_runner first to stream documents from a directory called data, and set up a Pathway VectorStoreServer.
Then run the frontend using 'streamlit run langchain_frontend.py' to set up the UI.

This is a basic RAG app which uses Gemini-1.5-Flash as the frontend.
Memory of chat is available, as it is fed to the model directly as part of the prompt(Gemini has a long context window, so this is feasible).

Users can upload documents as part of the prompt. These documents have the text read and attached to the prompt.
Direct uploads to the database are also supported.


testing.py is an alternative for langchain_frontend.py written using llamaindex features.
It can serve as a foundation for a better pipeline, as Pathway's integration with Llamaindex is more complete than that with Langchain.
