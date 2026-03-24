# technical-llamaindex
My technical workshop for the RSA group on llamaindex

## Getting started

### 1. Create and activate a virtual environment

```bash
# Create the venv (only needed once)
python -m venv .venv

# Activate it – macOS / Linux
source .venv/bin/activate

# Activate it – Windows (Command Prompt)
.venv\Scripts\activate.bat

# Activate it – Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

### 2. Install dependencies

With the venv active, install all required packages:

```bash
pip install -r requirements.txt
```

### 3. Set your OpenAI API key

Copy the example environment file and add your key:

```bash
cp .env.example .env
```

Open `.env` and replace `your-openai-api-key-here` with your actual key:

```
OPENAI_API_KEY=sk-...
```

> **Note:** `.env` is listed in `.gitignore` and will never be committed to the repository.

The notebooks load the key automatically via `python-dotenv`. Alternatively, you can set the variable in your shell before launching Jupyter:

```bash
export OPENAI_API_KEY=sk-...   # macOS / Linux
set OPENAI_API_KEY=sk-...      # Windows Command Prompt
```

### 4. Launch Jupyter

```bash
jupyter notebook
```
