# 💻 Installation: Getting Started

To run this project locally, follow these instructions:

## 1. **Clone this repository:**

```bash
git clone https://github.com/martaverfer/smart-book-recommender.git \
cd scripts
```

## 2. **Virtual environment:**

Create the virtual environment: 
```bash
python3 -m venv venv
```

Activate the virtual environment:

- For Windows: 
```bash
venv\Scripts\activate
```

- For Linux/Mac: 
```bash
source venv/bin/activate
```

To switch back to normal terminal usage or activate another virtual environment to work with another project run:
```deactivate```

## 3. **Install dependencies:**

```bash
pip install --upgrade pip; \
pip install -r requirements.txt
```

## 4. **Set Up OpenAI API Key:** 

To use the OpenAI API for generating book summaries, you'll need to set up your API key.

1. Go to [OpenAI](https://platform.openai.com/account/api-keys) to get yout API key.

2. Create the `.streamlit` folder in your app folder inside your project directory

```
your-project/
└── app/
    ├── app.py
    └── .streamlit/
```

3. Inside `.streamlit/`, create a `secrets.toml` file and add the following content:

```toml
OPENAI_API_KEY = "your-openai-api-key-here"
```

Replace `your-openai-api-key-here` with your actual OpenAI API key.

## 5. **Open the Jupyter notebooks to explore the analysis:**

```bash
cd scripts; \
1.data-exploration.ipynb
2.text-processing.ipynb
3.eda.ipynb
4.clustering.ipynb
```

This script will execute the analysis steps and produce the results.

## 5. **For Running Streamlit App**
```bash
cd app; \
streamlit run app.py
```