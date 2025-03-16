# MGTA415_Project **Predicting Chapter Ratings Using Sentiment Analysis**

## **Project Overview**
In today's novel market, many authors have transitioned from traditional full-book publications to **serialized storytelling**, updating their stories daily or weekly to maintain reader engagement. However, a major challenge is that **chapter-specific ratings are rarely assigned**, making it difficult for authors to gauge reader satisfaction in real time. This project aims to **predict chapter-level ratings** based on textual comments, leveraging sentiment analysis to provide insights into audience engagement.

By training a machine learning model on **Amazon book reviews**, which include both textual feedback and numerical ratings, we develop a model that can be applied to **Wuxiaworld chapter comments**—a dataset where direct ratings are absent. This approach allows us to estimate reader sentiment and provide actionable feedback for authors of serialized fiction.

---

## **Dataset**
### **Training Data**
- **Amazon Book Reviews Dataset** (from [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)):  
  - Contains **millions of book reviews**, each with both a **textual review** and a corresponding **numeric rating** (1-5 stars).
  - Helps in understanding how users express opinions and assigning sentiment-based numerical scores.
  - A **20,000-sample subset** is used to train the model efficiently.

### **Test Data**
- **Wuxiaworld Chapter Comments**:
  - Unlike Amazon reviews, Wuxiaworld comments are **shorter, more informal, and lack explicit ratings**.
  - These comments are used to test whether the model can generalize **from full-book reviews to serialized chapter feedback**.

---

## **Hypothesis & Methodology**
We hypothesize that **full-book reviews and chapter-specific comments share similar sentiment patterns** despite differences in reading habits and publishing structures. While serialized fiction encourages more frequent engagement, we believe the way readers express their opinions remains **consistent** across both formats.

To test this, we:
1. **Train a model** on Amazon book reviews to learn the relationship between textual features and ratings.
2. **Apply the model** to predict chapter ratings based on Wuxiaworld comments.
3. **Evaluate whether the predicted ratings align** with expected reader sentiment, helping authors gain valuable insights.

---

## **Implementation Summary (Jupyter Notebook)**
The implementation follows these key steps:

### **1. Data Collection**
- Extracted **book reviews and ratings** from the **Amazon Books dataset**.
- Gathered **chapter-specific comments** from **Wuxiaworld** as the test set.

### **2. Exploratory Data Analysis (EDA)**
- **Dataset Overview**:  
  - Used **20,000 book reviews** for training.
  - Wuxiaworld comments serve as the **real-world test case**.

### **3. Preprocessing & Feature Engineering**
- Cleaned and tokenized textual data, removing noise (e.g., stopwords, punctuation).
- Converted text into numerical representations using:
  - **TF-IDF (Term Frequency-Inverse Document Frequency)**
  - **Word embeddings** (Word2Vec, BERT)

### **4. Model Training**
- Trained a regression/classification model using Amazon book reviews, predicting ratings based on textual features.
- Experimented with different models:
  - **Linear Regression**
  - **Random Forest**
  - **Deep Learning (BERT, LSTM)**

### **5. Testing on Wuxiaworld Comments**
- Applied the trained model to **predict ratings** for Wuxiaworld chapter comments.
- Evaluated whether the **predicted scores matched expected sentiment patterns**.


