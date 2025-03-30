# **WorkWise AI**  
## **Team: The Backpropagators**

#### **Members:** Harshil Patel, Tanishq Jain and Priyank Shah

## **Project Overview**
In 2025, the world of AI agents is evolving, and we wanted to align our hackathon project with the latest trends. Inspired by ADP's requirement to build an AI-powered career guide, we present **WorkWise AI**—a **ReACT Agent**-based solution that can dynamically analyze real-time data based on the user's queries. The AI system aims to answer career-related questions, visualize compensation trends, and guide users with up-to-date, actionable insights.  

### **What does WorkWise AI do?**  
WorkWise AI is an AI-powered career assistant that:
- **Analyzes compensation trends** – Understands salary growth, skill valuation, and career progression.
- **Answers career-impacting questions** – Provides strategic insights using natural language.
- **Visualizes opportunities** – Transforms raw data into actionable insights.
- **Simulates career paths** – Predicts how skill upgrades or role changes affect compensation.

### **How does it work?**
The ReACT Agent leverages cutting-edge AI tools to perform real-time data analysis based on user queries. For example, if a user asks, "What is the average salary provided by Google for Software Engineering?", the AI performs the necessary data analysis to provide the most relevant and up-to-date information. For the scope of this hackathon, data analysis is focused on salary information and company-related queries.

---

## **Technologies Used**
- **ReACT Agent Framework**: Powering the dynamic, real-time query handling.
- **LangChain**: A framework for building applications using large language models (LLMs).
- **Plotly**: For generating dynamic and interactive visualizations.
- **FastAPI**: For building the backend API.
- **Groq AI**: For executing complex queries.
- **Pandas**: For handling and analyzing datasets.
- **Tavily API**: For searching and gathering general information from various sources.
- **Python**: Primary language used for backend development.
- **.env**: Environment variables to securely store API keys.

---

## **Features**
- **Real-time Data Analysis**: The AI can analyze and respond to queries related to salary trends, company insights, and career advice.
- **Salary Analysis**: Provides compensation insights and comparisons for specific roles and companies.
- **Company Information**: Displays employee sentiments and insights about a company's work culture, benefits, and more.
- **Visualization**: Dynamic, actionable visualizations (e.g., salary trends, job growth, etc.).
- **Flexible Integration**: Easily extensible to handle a wide variety of career-related queries beyond salaries and company data.

---

## **How to Use**

### **1️⃣ Clone the Repository**
```sh
git clone https://github.com/your-repo/workwise-ai.git
cd workwise-ai
```

### **2️⃣ Install Dependencies**
```sh
pip install -r requirements.txt
```

### **3️⃣ Set Up Environment Variables**
```sh
GROQ_API_KEY=your_groq_api_key
TAVILY_API_KEY=your_tavily_api_key
```

### **4️⃣ Run the API**
```sh
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### **API Endpoints**

#### **API BASE:** : https://3319-2601-8c-4e80-ab40-585-a716-4667-7848.ngrok-free.app

#### **/get_response (POST)**
- Description: Processes the user's query and provides career-related insights along with any relevant visualizations.

## **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Video Link :
    <iframe style="border:none;width:100%;height:100%;" src="[https://www.youtube.com/embed/VIDEO_I](https://youtu.be/omylt3E9Z1E)" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>


