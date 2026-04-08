## ZZY LLM – Local Chat UI for Ollama

ZZY LLM is a lightweight desktop chat interface built with PySide6 that connects directly to your local Ollama server.
It lets you experiment with locally-run LLMs such as qwen3-vl, gpt-oss, or any other Ollama model — featuring chat history, file attachment, and experiment routing support.

### 🧑‍💻 Developer Setup
1. Environment Setup

Ensure Poetry is installed and available:

poetry --version
pip install poetry


Activate the Poetry environment:

poetry env activate

2. Build and Release the UI App

Run the PowerShell script to build and release for Python 3.13:

.\build.ps1 -Task release-ui -PyVersions @('3.13')

3. Local Development

Use VS Code for debugging:

Open the project folder in VS Code

Select Python: Debug module zzy_llm.main from the Run & Debug dropdown

Start the debugger to launch the app locally

### 🧠 Ollama Setup
1. Download & Install

Download Ollama from the official site:
👉 https://ollama.com/download

2. Verify Installation

Open Command Prompt and run:

ollama


If the CLI help appears, Ollama is installed correctly.

3. Choose a Model

Browse available models at
👉 https://ollama.com/library

Examples:

qwen3-vl:latest

gpt-oss:latest

4. Pull a Model
ollama pull qwen3-vl:latest

5. Run a Model
ollama run qwen3-vl

⚙️ Helpful Ollama Commands
Description	Command
Show all available commands	ollama
List downloaded models	ollama list
Download a model	ollama pull <model-name>
Stop a running model	ollama stop <model-name>
Stop all models	ollama stop all