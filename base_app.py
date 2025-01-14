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
news_vectorizer = open("resources/vector3.pkl","rb")
tweet_cv = joblib.load(news_vectorizer) # loading your vectorizer from the pkl file

# Load your raw data
raw = pd.read_csv("resources/train.csv")



logo_image = "resources/imgs/logo1.png"
# The main function where we will build the actual app
st.sidebar.image(logo_image, width=200)
def main():
	"""Tweet Classifier App with Streamlit """

	# Creates a main title and subheader on your page -
	# these are static across all pages

	st.title("TwitterInsightify Classifier")
	st.subheader("Climate change tweet classification")

	# Creating sidebar with selection box -
	# you can create multiple pages this way
	with st.sidebar:
		selection = option_menu("Main Menu", ["Model", "More Info", "Development team","How to Use"], 
        icons=['house', 'pie-chart', 'people-fill', 'envelope'], menu_icon="cast", default_index=0)
	#options = ["Prediction", "Information", "Development team"]
	#selection = st.sidebar.selectbox("Choose Option", options)

	# Building out the "Information" page





	
	if selection == "More Info":
		st.subheader("About the App")
		# You can read a markdown file from supporting resources folder
	
		st.markdown("Hello, welcome to our TwitterInsightify Classifier app! With companies striving to minimize their impact on nature, our app is designed to tackle a pressing issue – understanding public perceptions of climate change. Through the power of Machine Learning, we've created a tool that, in the future, will assist companies in grasping what people will think and feel about their products.")
		st.markdown("By analyzing tweet data, our app provides insightful classifications, empowering businesses to tailor their strategies effectively. Join us as we explore the significance of this task, the data-driven approach we've taken, and the potential impact on shaping marketing strategies across diverse demographics and geographical regions. Let's navigate through the world of environmental consciousness together.")
		
		st.subheader("Different Sentiments")
		mycol1,mycol3 = st.columns(2)
		with mycol1:
			st.success("2")
			st.info("1")
			st.warning("0")
			st.error("-1")
		with mycol3:
			st.markdown("News: the tweet links to factual news about climate change")
			st.markdown("Pro: the tweet supports the belief of man-made climate change")
			st.markdown("Neutral: the tweet neither supports nor refutes the belief of man-made climate change")
			st.markdown("Anti: the tweet does not believe in man-made climate change")
		if st.checkbox('Show raw data'): # data is hidden if box is unchecked
			
			st.write(raw[['sentiment', 'message']].head()) # will write the df to the page

			opt = st.radio('Plot  type:',['Bar', 'Pie', 'Word Cloud'])
			if opt=='Bar':
				st.markdown('<h3>Show sentiment occurance dataset</h3>',unsafe_allow_html=True)
				xx = raw['sentiment'].value_counts()
				st.bar_chart(xx)
			elif opt =="Pie":
				st.markdown('<h3>Pie chart for percentage of each sentiment on dataset</h3>',unsafe_allow_html=True)
				fig1, ax1 = plt.subplots()
				ax1.pie(raw['sentiment'].value_counts(),labels = ["Pro","News","Neutral","Anti"], autopct='%1.1f%%',shadow=True, startangle=90)
				ax1.axis('equal')
				ax1.set_facecolor("black")  # Equal aspect ratio ensures that pie is drawn as a circle.
				ax1.legend()
				fig1.patch.set_alpha(0)
				ax1.xaxis.label.set_color('red')
				st.pyplot(fig1)
				
		
			else:
				st.set_option('deprecation.showPyplotGlobalUse', False)
				st.markdown('<h3>Word Cloud for how frequently words show up on all tweets.</h3>',unsafe_allow_html=True)
				allwords = ' '.join([msg for msg in raw['message']])
				WordCloudtest = WordCloud(width = 800, height=500, random_state = 21 , max_font_size =119).generate(allwords)
				
				plt.imshow(WordCloudtest, interpolation = 'bilinear')
				
				plt.axis('off')
				st.pyplot(plt.show())



	if selection == "Development team":
		st.title("Meet our team")
		st.title("")
		st.divider()
		col1, mid, col2 = st.columns([80,20,80])
		with col2:
			st.subheader("Nontokozo Ndlovu - Lead Engineer")
			st.write("Nontokozo Ndlovu has worked as a Project Manager, Product Manager, Systems and Production developer. When she is not coding he enjoys watching sport on television.")
		with col1:
			st.image('nonto.jpg', width=380)
			
		st.divider()
		col1, mid, col2 = st.columns([80,20,80])
		with col1:
			st.subheader("Siyabonga Mkhize - Data Scienstist")
			st.write("Siyabonga has worked as a data scientist for various companies including Netflix and Apple to name a few. In his spare time he likes to spend time with family and watch football")
		with col2:
			st.image('sya.jpg', width=300)		

		st.divider()
		col1, mid, col2 = st.columns([80,20,80])
		with col2:
			st.subheader("Amanda Mtshali- Machine learning engineer")
			st.write("She has designed predicted models for companies such as FNB and BMW. One of my project was creating a chatbot with Python's NTLK library.She is a fitness fanatic and loves dancing ")
		with col1:
			st.image('anele.jpg', width=380)

		st.divider()

		col1, mid, col2 = st.columns([80,20,80])
		with col1:
			st.subheader("Tamika Gavington- Machine learning engineer")
			st.write("She has designed predicted models for companies such as FNB and BMW. One of my project was creating a chatbot with Python's NTLK library.She is a fitness fanatic and loves dancing ")
		with col2:
			st.image('tamika.jpg', width=300)
		
		st.divider()

		col1, mid, col2 = st.columns([80,20,80])
		with col2:
			st.subheader("Saneliswa Ndlovu- Machine learning engineer")
			st.write("Saneliswa Ndlovu brings a unique blend of technical expertise to the table. As a machine learning engineer, she excels at designing and implementing predictive models. Beyond web development, Saneliswa is a creative writer who enjoys expressing herself through storytelling.")
		with col1:
			st.image('sanelisiwe.jpg', width=300)


	# Building out the predication page
	if selection == "Model":
		st.info("Prediction with ML Models")
		# Creating a text box for user input
		model_type = st.radio('Model  type:',['Logistic Regression', 'LR-L1/L2 Penalty','LR-lbfgs olver'])
		tweet_text = st.text_area("Enter Text","Type Here")

		if st.button("Classify"):
			# Transforming user input with vectorizer
			vect_text = tweet_cv.transform([tweet_text]).toarray()
			# Load your .pkl file with the model of your choice + make predictions
			# Try loading in multiple models to give the user a choice
			if model_type=='Logistic Regression':
				predictor = joblib.load(open(os.path.join("resources/model_logistic.pkl"),"rb"))

			elif model_type=='LR-L1/L2 Penalty':
				predictor = joblib.load(open(os.path.join("resources/model_linear_2.pkl"),"rb"))
			elif model_type=='LR-lbfgs olver':
				predictor = joblib.load(open(os.path.join("resources/model_linear_3.pkl"),"rb"))


				
			prediction = predictor.predict(vect_text)

			# When model has successfully run, will print prediction
			# You can use a dictionary or similar structure to make this output
			# more human interpretable.

			prediction_dic =  {-1:"Anti: the tweet does not believe in man-made climate change", 0:"Neutral: the tweet neither supports nor refutes the belief of man-made climate change",
			1:"Pro: the tweet supports the belief of man-made climate change", 2:"News: the tweet links to factual news about climate change"}
			st.success("Text Categorized as: {}".format(prediction_dic[prediction[0]]))

		# Building the "How to Use" page
	if selection == "How to Use":
		st.subheader("Follow these easy steps:")
		st.write("Step 1: We have options on the left")
		st.write("Step 2: Choose the 'Prediction' option")
		st.write("Step 3: Then choose a model from the available choices")
		st.write("Step 4: Enter a tweet in the text area")
		st.write("Step 5: Click Classify!")

# Required to let Streamlit instantiate our web app.  
if __name__ == '__main__':
	main()
