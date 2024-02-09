import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder

#-------------------------------------------------------------------------------#
#------------------------------Funtions-----------------------------------------#
#-------------------------------------------------------------------------------#
"""
    1. import_data(): funtion that import the data
    2. duplicate_split(): Funtion that search if there duplicate values.
        Futhermore, create a variable split  to  cabin and add three new columns that get the data of the split.
        Finally, delete columns like Passager ID, Name, Cabin. Data that aren´t going to go inside of the analysis
    3. delete_data_null(): Funtion that search null values and are replace with the mean or  value repeated most often
    4. convert_int(): Funtion that look at that the values are int type
    5. chance_categorical(): Funtion that is responsible of do the change categorical using library  labelencoder
    6. create_db(): funtion that is responsible of create data basee using the funtions previous
"""
def import_data(data):
    file_path = os.path.abspath(data)
    df=pd.read_csv(file_path)    
    return df


def duplicate_split(df):
    #Drop duplicate rows
    df.drop_duplicates(inplace = True)
    
    #split colum cabin to create three news columns
    df.insert(4,"Deck", value='\\N' )
    df.insert(5,  "Cabin_num", value='\\N')
    df.insert(6,  "Side", value='\\N')
    
    #making split to column "Cabin"
    df[["Deck", "Cabin_num", "Side"]] = df["Cabin"].str.split("/", expand=True)

    #delate column "Cabin" and "Name" of the database
    df.drop(columns=['PassengerId','Name', 'Cabin'], inplace=True)

    return df

def delete_data_null(df):
    #calculate mean to numeric data
    df['Age'].fillna(df['Age'].mean(), inplace=True)
    df['RoomService'].fillna(df['Age'].mean(), inplace=True)
    df['FoodCourt'].fillna(df['FoodCourt'].mean() , inplace=True)
    df['ShoppingMall'].fillna(df['ShoppingMall'].mean(), inplace=True)
    df['Spa'].fillna(df['Spa'].mean(), inplace=True)
    df['VRDeck'].fillna(df['VRDeck'].mean(), inplace=True)

    
    #change null values to the value that is repeated most often
    for column in df.columns:
        df[column]=df[column].fillna(df[column].value_counts().idxmax())

    return df

def convert_int(df):
    #convert data type float to int
    df['Age']=df['Age'].astype('int32')
    df['RoomService']=df['RoomService'].astype('int32')
    df['FoodCourt']=df['FoodCourt'].astype('int32')
    df['ShoppingMall']=df['ShoppingMall'].astype('int32')
    df['Spa']=df['Spa'].astype('int32')
    df['VRDeck']=df['VRDeck'].astype('int32')
    df['Cabin_num']=df['Cabin_num'].astype('int32')

    return df

def chance_categorical(df, type_):
  #it use labelencoder to encoder the columns object type
  for dt in df.columns:      
        # Inicializar el codificador de etiquetas
        label_encoder = LabelEncoder()
        df[dt]=label_encoder.fit_transform(df[dt])
  return df

def upload_y_test(path):
    
    df=import_data(path)
    df=chance_categorical(df)
    return df


############################################################################################################
#------------------------------------------Funtion to create DB---------------------------------------------#
#############################################################################################################
def create_db(path):
    df=import_data(path)
    df=duplicate_split(df)
    df=delete_data_null(df)    
    df=convert_int(df)
    df=chance_categorical(df, 'object')
    return df

def input_data(df):
    #split colum cabin to create three news columns
    
    df=duplicate_split(df)
    df=delete_data_null(df)    
    df=convert_int(df)
    df=chance_categorical(df, 'object')
    return df

