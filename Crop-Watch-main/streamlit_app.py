import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model(r"C:\Users\dewan\Desktop\plant_disease_detection copy\trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(128,128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) #convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

def weed_model_prediction(test_image_path):
    # Load the model outside the function to avoid redundant loading
    weed_model = tf.keras.models.load_model(r"C:\Users\dewan\Desktop\weed detection model\RCNN_crop_weed_classification_model.h5")
    
    # Load and preprocess the image
    weed_image = tf.keras.preprocessing.image.load_img(test_image_path, target_size=(224, 224))
    weed_input_arr = tf.keras.preprocessing.image.img_to_array(weed_image)
    weed_input_arr = np.expand_dims(weed_input_arr, axis=0)  # Convert single image to batch by adding an extra dimension
    
    # Predict
    predictions = weed_model.predict(weed_input_arr)
    
    # Return the predicted class label index
    return np.argmax(predictions)



def seed_model_prediction(seed_test_image_path):
    # Load the SavedModel (.pb format)
    loaded_model = tf.saved_model.load(r"C:\Users\dewan\Desktop\Soybean Seeds - Copy\trained_soyabean_seed_model")
    
    # Preprocess the image
    seed_image = tf.keras.preprocessing.image.load_img(seed_test_image_path, target_size=(128, 128))
    seed_input_arr = tf.keras.preprocessing.image.img_to_array(seed_image)
    seed_input_arr = np.expand_dims(seed_input_arr, axis=0)  # Convert single image to batch by adding an extra dimension
    
    # Convert the input array to a tensor
    seed_input_tensor = tf.convert_to_tensor(seed_input_arr, dtype=tf.float32)
    
    # Perform inference
    predictions = loaded_model(seed_input_tensor)
    
    # Get the predicted class label index
    predicted_index = np.argmax(predictions)
    
    return predicted_index






#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About","Disease Recognition","Weed Detection","Seed Detection"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path,use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

#About Project
elif(app_mode=="About"):
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on this github repo.
                This dataset consists of about 87K rgb images of healthy and diseased crop leaves which is categorized into 38 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purpose.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

#Prediction Page
elif(app_mode=="Disease Recognition"):
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
                    'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
                    'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
                    'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                    'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
                    'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
                    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
                    'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
                    'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_index]))

elif app_mode == "Weed Detection":
    st.header("Weed Detection")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=300, use_column_width=True)
        # Predict button
        if st.button("Predict"):
            st.spinner("Predicting...")
            result_index = weed_model_prediction(test_image)
            # Reading Labels
            class_name = ['background', 'crop', 'weed']
            st.success("Model predicts it's a {}".format(class_name[result_index]))

elif app_mode == "Seed Detection":
    st.header("Seed Detection")
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None:
        if st.button("Show Image"):
            st.image(test_image, width=300, use_column_width=True)
        # Predict button
        if st.button("Predict"):
            st.spinner("Predicting...")
            # Convert the uploaded image to bytes
            image_bytes = test_image.read()
            # Save the bytes as an image file
            with open("temp_image.jpg", "wb") as f:
                f.write(image_bytes)
            # Perform prediction
            result_index = seed_model_prediction("temp_image.jpg")
            # Reading Labels
            class_name = ['Broken soybeans', 'Immature soybeans', 'Intact soybeans','Skin-damaged soybeans','Spotted soybeans']
            st.success("Model predicts it's a {}".format(class_name[result_index]))



        
