# 🎮 Game Concept Market Analyzer

A data-driven dashboard for exploring Steam game markets — combining **hybrid retrieval (FAISS + TF-IDF)**, **statistical analytics**, and an **AI-powered publishing advisor**.

🧠 Built with **Python · Streamlit · FAISS · scikit-learn · Groq LLM API**

---

## 🌐 Live Demo

> 🟢 **Try it here:** [https://YOUR-APP-NAME.streamlit.app](https://YOUR-APP-NAME.streamlit.app)  
> *(Public interactive demo — no installation or API key needed.)*

---

## 🧩 Key Features

- **📊 Market Explorer** – Analyze existing Steam games by tags, price, and positive-review ratio.  
- **🧪 What-If Simulator** – Describe your own game concept and compare it with similar titles.  
- **🤖 AI Advisor** – Generate a data-aware publishing report powered by the Groq LLM API.  
- **📈 Statistical Depth** – Includes regressions, sales proxies, and pricing-to-sales visuals.

---

## ⚙️ How to Run Locally (Developers)

```bash
# 1️⃣ Create a virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2️⃣ Install dependencies
pip install -r requirements.txt

# 3️⃣ Create a .env file
cp .env.example .env
# → Then open it and add your own GROQ_API_KEY

# 4️⃣ Launch
streamlit run src/dashboard/app.py
