from fastapi import FastAPI, HTTPException
from pydantic import BaseModel 
from sklearn.preprocessing import LabelEncoder
import pickle
import uvicorn
import streamlit as st
import requests
from typing import List
import asyncio
import pandas as pd

class ScoringItem(BaseModel):
    name:str
    HomePlanet: int      
    CryoSleep: int      
    Deck: int
    Cabin_num: int
    Side:  int           
    Destination: int     
    Age : int           
    VIP: int             
    RoomService: int     
    FoodCourt: int       
    ShoppingMall: int    
    Spa : int            
    VRDeck: int

app = FastAPI()
@app.get("/user")
async def user():
    return {'Message': 'Welcome to this API to predict machine learning model'}

#uvicorn main:app --reload
#Streamlit run main.py   
@app.post("/randomforest")
async def randomforest(datos: List[ScoringItem]):
    #Load pickle file that contains the predict model
    rf_pickle=open('randomforest.pkl', 'rb')
    rfc=pickle.load(rf_pickle)
    rf_pickle.close()

    #list that contains the data from list[Scoringitem]. This are nested lists
    input_data = [
        [item.HomePlanet, item.CryoSleep, item.Deck, item.Cabin_num,
         item.Side, item.Destination, item.Age, item.VIP, item.RoomService,
         item.FoodCourt, item.ShoppingMall, item.Spa, item.VRDeck]
        if isinstance(item, ScoringItem)
        else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for item in datos
    ]

    # Apply predict model 
    new_predictions = rfc.predict(input_data)
    if new_predictions[0]==0:
        status="The passager was salved"
    else:
        status="The passager is dead"
    
    return {"Predict model": "Random Forest", "user name":datos.name, "Prediction": status}



@app.post("/gausnb")
async def gausnb(datos: List[ScoringItem]):
    #Load pickle file that contains the predict model
    gnb_pickle=open('gaussianNB.pkl', 'rb')
    gnbc=pickle.load(gnb_pickle)
    gnb_pickle.close()

    #list that contains the data from list[Scoringitem]. This are nested lists
    input_data = [
        [item.HomePlanet, item.CryoSleep, item.Deck, item.Cabin_num,
         item.Side, item.Destination, item.Age, item.VIP, item.RoomService,
         item.FoodCourt, item.ShoppingMall, item.Spa, item.VRDeck]
        if isinstance(item, ScoringItem)
        else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for item in datos
    ]
    # Apply predict model 
    new_predictions = gnbc.predict(input_data)
    if new_predictions[0]==0:
        status="The passager was salved"
    else:
        status="The passager is dead"
    
    return {"Predict model": "GaussianNB", "user name":datos.name, "Prediction": status}

@app.post("/logisticModel")
async def logisticModel(datos: List[ScoringItem]):
    #Load pickle file that contains the predict model
    lgr_pickle=open('logisticregression.pkl', 'rb')
    lgr=pickle.load(lgr_pickle)
    lgr_pickle.close()

    #list that contains the data from list[Scoringitem]. This are nested lists
    input_data = [
        [item.HomePlanet, item.CryoSleep, item.Deck, item.Cabin_num,
         item.Side, item.Destination, item.Age, item.VIP, item.RoomService,
         item.FoodCourt, item.ShoppingMall, item.Spa, item.VRDeck]
        if isinstance(item, ScoringItem)
        else [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for item in datos
    ]
    # Apply predict model 
    new_predictions = lgr.predict(input_data)
    if new_predictions[0]==0:
        status="The passager was salved"
    else:
        status="The passager is dead"
    
    return {"Predict model": "Logistic Regression", "user name":datos.name, "Prediction": status}

#####################################################################################################
#---------------------------------Streamlit----------------------------------------------------------
#####################################################################################################
async def main():
    # Application page configuration
    st.set_page_config(page_title="ML Classification API ", layout="wide")

    # Application title
    st.title("ML Classification API for the Titanic Ship")

    # getting  user input data  
    #Allow user upload data
    upload_file=st.sidebar.file_uploader("Upload you csv file", type=["csv"])
    if upload_file is not None:
        input_df=pd.read_csv(upload_file)
        input_df = input_df.applymap(lambda x: LabelEncoder().fit_transform([x])[0])
    else:
        
            #create the data selection controls for the user to enter the data
        name_ = st.text_input("Enter your name", key="name")
        hp = st.sidebar.slider('Homeplanet. Europa: 0, Earth: 1, Mars: 2', 0, 1, 2, key="hp")
        cs = st.sidebar.slider('CryoSleep. False: 0, True: 1', 0, 1, key="cs")
        deck = st.sidebar.slider('Deck. A:0, B:1, C:2, D:3, F:4, G:5', 0, 5, key="deck")
        nc = st.number_input("Enter the cabin number", key="nc")
        side = st.sidebar.slider('Side. P: 0, S1:1', 0, 1, key="side")
        dt = st.sidebar.slider('Destination. PSO J318.5-22: 0, TRAPPIST-1e: 1, 55 Cancri e: 2', 0, 1, 2, key="dt")
        age = st.number_input("Enter your age", key="age")
        vip = st.sidebar.slider("VIP False: 0, True:1", 0, 1, key="vip")
        room_service = st.number_input("Enter the Room Service", key="room_service")
        food_court = st.number_input("Enter the Fourt Court", key="food_court")
        shopping_mall = st.number_input("Ingrese Shopping mall", key="shopping_mall")
        spa = st.number_input("Enter the spa number", key="spa")
        vr_deck = st.number_input("Enter the VRDeck number", key="vr_deck")

            #creation dataframe dictionary
        features = {
                    "name": name_,
                    "HomePlanet": hp,
                    "CryoSleep": cs,
                    "Deck": deck,
                    "Cabin_num": nc,
                    "Side": side,
                    "Destination": dt,
                    "Age": age,
                    "VIP": vip,
                    "RoomService": room_service,
                    "FoodCourt": food_court,
                    "ShoppingMall": shopping_mall,
                    "Spa": spa,
                    "VRDeck": vr_deck
                    }
            
        input_df=pd.DataFrame([features], index=[0])
            #it see if the variable next are of the type int
            # Convertir las columnas relevantes a tipos de datos enteros si es necesario
        int_columns = ["Age", "RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck", "Cabin_num"]
        input_df[int_columns] = input_df[int_columns].astype('int32')
        #input_df = input_df.to_dict(orient='dict')
             
    
    st.subheader("Data entered by the user")
    # Botón para enviar los datos a los modelos de clasificación

    if upload_file is not None:
        st.write(input_df)
    else:
        st.write("waiting for csv data to be uploaded")
        st.write(input_df)


    if st.button("Classify"):
        data=input_df
        #data = input_df.to_dict(orient='records')
        
        # Crear una lista de tareas asíncronas para los modelos de clasificación
        tasks = [
            asyncio.create_task(randomforest(data)),
            asyncio.create_task(gausnb(data)),
            asyncio.create_task(logisticModel(data))
        ]

        # Esperar a que todas las tareas se completen
        resultados = await asyncio.gather(*tasks)

        # Mostrar los resultados de los modelos de clasificación
        st.write("Results:")
        st.write("randomforest:", pd.DataFrame(resultados[0]))
        st.write("GaussianNB:", pd.DataFrame(resultados[1]))
        st.write("Logistic Model:", pd.DataFrame(resultados[2]))

if __name__ == "__main__":
    #uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
    asyncio.run(main())