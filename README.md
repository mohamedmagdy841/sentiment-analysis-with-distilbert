### Sentiment Analysis for Social Media Apps Reviews

<p align="center">
  <img width="500" src="https://github.com/user-attachments/assets/b9869141-d218-4ea8-b003-4d33b051dd98">
</p>

This project aims to classify user sentiments into positive, negative, or neutral categories. Each part of this project demonstrates how to:

1. **Scrape Data using `google-play-scraper`**
2. **Visualize and Explore the Data**
3. **Preprocess the Data**
4. **Get the Tokenizer of a Pre-trained Model**
5. **Fine-tune the Model to the Specific Problem**
6. **Train the Model**
7. **Evaluate the Model**
8. **Deploy the Model**

#### 1. Scrape Data using `google-play-scraper`
- The dataset was scraped from Google Play social media apps by collecting user reviews. The `google-play-scraper` library was used for this purpose, allowing for automated data collection from various apps.

#### 2. Visualize and Explore the Data
- Visualization tools such as Matplotlib and Seaborn were utilized to explore the dataset, providing insights into the distribution of reviews, the frequency of positive, negative, and neutral sentiments, and other relevant patterns.

#### 3. Preprocess the Data
- Data preprocessing involved cleaning the text, removing stopwords, handling missing values, and tokenizing the reviews. This step ensures that the data is in the best possible shape for model training.

#### 4. Get the Tokenizer of a Pre-trained Model
- A tokenizer from a pre-trained model (such as DistilBERT) was employed to convert the textual data into a format suitable for model input.

#### 5. Fine-tune the Model to the Specific Problem
- The pre-trained model was fine-tuned on the specific dataset to adapt its understanding and improve its performance on classifying social media app reviews.

#### 6. Train the Model
- The model was trained using the preprocessed data, with hyperparameters tuned to achieve optimal performance.

#### 7. Evaluate the Model
- Model evaluation was conducted using metrics such as accuracy, precision, recall, and F1-score to assess its performance on the sentiment classification task.

#### 8. Deploy the Model
- Finally, the trained model was deployed using a framework like Gradio, making it accessible for real-time sentiment analysis of social media app reviews.

By following these steps, the project successfully classifies user sentiments from social media app reviews, providing valuable insights into user opinions and feedback.

The dataset was scraped from Google Play social media apps by collecting user reviews.
<p align="center">
  <img width="800" src="https://github.com/mohamedmagdy841/sentiment-analysis-with-distilbert/assets/64127744/589ca7fc-d036-4239-a3bf-c22ccc1f4c93">
</p>

## BERT vs DistilBERT
DistilBERT is 60% faster than BERT. Additionally, it has 44 million fewer parameters and is 40% smaller in total than BERT. Despite this reduction, it retains 97% of BERT's performance.

### You can try the model here : https://huggingface.co/spaces/mmagdy841/sentiment-analysis-with-distilbert

## Demo
https://github.com/mohamedmagdy841/sentiment-analysis-with-distilbert/assets/64127744/9d25da2f-b862-4896-abfe-92ce8118ddbf




