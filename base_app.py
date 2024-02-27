"""

    Simple Streamlit webserver application for serving developed classification
	models.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.
    ---------------------------------------------------------------------

    Description: This file is used to launch a minimal streamlit web
	application. You are expected to extend the functionality of this script
	as part of your predict project.

	For further help with the Streamlit framework, see:

	https://docs.streamlit.io/en/latest/

"""
# Streamlit dependencies
import streamlit as st
import joblib,os

# Data dependencies
import pandas as pd
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from wordcloud import WordCloud
# Vectorizer
news_vectorizer = open("resources/vector.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")

# The main function where we will build the actual app
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages
	st.title("Tweet Classifer")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	with st.sidebar:
		selection = option_menu("Main Menu", ["Prediction", "Information", "Development team"], 
        icons=['house', 'pie-chart', 'people-fill', 'envelope'], menu_icon="cast", default_index=1)
	#options = ["Prediction", "Information", "Development team"]
	#selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page
	if selection == "Information":
		st.info("General Information")
		# You can read a markdown file from supporting resources folder
		st.markdown("Some information here")

		st.subheader("Raw Twitter data and label")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			st.write(raw[['sentiment', 'message']]) # will write the df to the page

	if selection == "Development team":
		st.title("Meet our team")
		st.title("")
		col1, mid, col2 = st.columns([80,10,80])
		with col2:
			st.subheader("Nontokozo Ndlovu - Lead Engineer")
			st.write("Nontokozo Ndlovu has worked as a Project Manager, Product Manager, Systems and Production developer. When she is not coding he enjoys watching sport on television.")
		with col1:
			st.image('nonto.jpg', width=380)
		col1, mid, col2 = st.columns([80,10,80])
		with col1:
			st.subheader("Siyabonga Mkhize - Data Scienstist")
			st.write("Siyabonga has worked as a data scientist for various companies including Netflix and Apple to name a few. In his spare time he likes to spend time with family and watch football")
		with col2:
			st.image('anele.jpg', width=380)		


		col1, mid, col2 = st.columns([80,10,80])
		with col2:
			st.subheader("Amanda Mtshali- Machine learning engineer")
			st.write("She has designed predicted models for companies such as FNB and BMW. One of my project was creating a chatbot with Python's NTLK library.She is a fitness fanatic and loves dancing ")
		with col2:
			st.image('anele.jpg', width=380)

		col1, mid, col2 = st.columns([80,10,80])
		with col2:
			st.subheader("Tamika Gavington- Machine learning engineer")
			st.write("She has designed predicted models for companies such as FNB and BMW. One of my project was creating a chatbot with Python's NTLK library.She is a fitness fanatic and loves dancing ")
		with col2:
			st.image('anele.jpg', width=380)

		col1, mid, col2 = st.columns([80,10,80])
		with col2:
			st.subheader("Saneliswa Ndlovu- Machine learning engineer")
			st.write("She has designed predicted models for companies such as FNB and BMW. One of my project was creating a chatbot with Python's NTLK library.She is a fitness fanatic and loves dancing ")
		with col2:
			st.image('anele.jpg', width=380)


	# Building out the predication page
	if selection == "Prediction":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			predictor = joblib.load(open(os.path.join("resources/mlr_model.pkl"),"rb"))
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.

			prediction_dic =  {-1:"Anti: the tweet does not believe in man-made climate change", 0:"Neutral: the tweet neither supports nor refutes the belief of man-made climate change",
			1:"Pro: the tweet supports the belief of man-made climate change", 2:"News: the tweet links to factual news about climate change"}
			st.success("Text Categorized as: {}".format(prediction_dic[prediction[0]]))

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
