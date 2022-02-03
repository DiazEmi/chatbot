import pymongo

from preproc import PreProc
from Vec_carrera import vec_carrera
from Vec_w5 import vec_w5
from Vec_intencion_sub import vec_intents
import pandas as pd

n=0
# Conexion con Base de Datos
client = pymongo.MongoClient("mongodb://IES:IES@cluster0-shard-00-00.a4r4t.mongodb.net:27017,cluster0-shard-00-01.a4r4t.mongodb.net:27017,cluster0-shard-00-02.a4r4t.mongodb.net:27017/pln?ssl=true&replicaSet=atlas-7k3saa-shard-0&authSource=admin&retryWrites=true&w=majority")
mydb = client["pln"]
mycol = mydb["iefi"]

for data in mycol.find():
    n=int(data['_id'])

#Creacion del Corpus a Procesar.

n = n+1

subID = 0
s=True
while (s==True):
    entendio=True
    preguntaI = input("Hola, bienvenido en que te puedo ayudar?: (Para salir presione 1)\n\n>")
    if preguntaI == '1':
        break
    pregunta = PreProc(preguntaI)
    carrera = str(vec_carrera(pregunta))
    w5 = str(vec_w5(pregunta))
    intents = str(vec_intents(pregunta))
    (inten,subinten) = intents.split('#')
    
    #type(intents)
    
    #print(carrera)
    #print(w5)
    #print(inten)
    #print(subinten)
    
    Respuestas = pd.read_csv('tbl_respuestas.csv')
    
    
    Respuestas2 = Respuestas[(Respuestas.Intencion==inten) \
                           & (Respuestas.SubIntencion==subinten) \
                           & (Respuestas.Carrera==carrera) \
                           & (Respuestas.w5==w5)]
    if len(Respuestas2)==0:
        respuesta = "No entendí su pregunta, podria usted reformularla?"
        entendio=False
    else:
        respuesta = Respuestas2['Respuesta'].values[0]
    
    print(respuesta)
    subID = subID+1         
    i = {'_id':n,'N_Preg':subID,'Pregunta_Inicial':preguntaI,'Pregunta_Procesada':pregunta,'carrera':carrera,'w5':w5,'intencion':inten,'subintencion':subinten,'Respuesta':respuesta,'Entendio':entendio}          
    x = mycol.insert_one(i)         
    