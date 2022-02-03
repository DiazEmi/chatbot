# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 00:40:51 2021

@author: fabri
"""

# Codificación
# -*- coding: utf-8 -*-

# Bibliotecas que usaremos
import numpy as np
import pandas as pd
import sklearn
import nltk
import re
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV



def vec_w5(new_doc):
    
    # Spacy
    import spacy
    nlp=spacy.load('es_core_news_lg')
    
    # Stemmer
    from nltk.stem import SnowballStemmer
    spanish_stemmer = SnowballStemmer('spanish')
    
    # Levantamos la lista de StopWords
    f = open('stopwords.txt', 'r', encoding='utf8')
    stopwords = f.read().split('\n')
    f.close()
    
    
    def PreProcesar(Corpus, POS=False, Lema=True, Stem=True):
        """
        Recibe como parámetro una lista de cadenas. Cada elemento es un documento (oración) del corpus
        a procesar.
        Devuelve una lista de documentos con sus palabras filtradas y transformadas: Cada elemento de 
        la lista devuelta es la cadena correspondiente a la oración original pero transformada según lo
        indiquen los parámetros:
        - POS: Si recibe True agrega a la palabra su POS Tagging. Por ejemplo "inteligencia" se convierte 
               en "inteligencia_noun", "artificial" se convierte en "artificial_adj"
        - Lema: Cada palabra se lematiza previamente.
        - Stem: Se realiza stemming de cada palabra, dejando su raíz y eventualmente su POS.
        """
        

        # Generar una lista de documentos de spacy para tratar el POS Tagging y la Lematización
        docs=[]
        for oracion in Corpus:
            docs.append(nlp(oracion.lower())) #La lematización funciona mejor en minúsculas
        
        # Crear una lista de oraciones, donde cada elemento es una lista de palabras.
        # Cada palabra está definida por una tupla (Texto, POSTag, Lema)
        # Se omiten los tokens que son identificados como signos de puntuación
        oraciones=[]
        for doc in docs:
            oracion=[]
            for token in doc:
                if token.pos_ != 'PUNCT':
                    oracion.append((token.text, token.pos_, token.lemma_))
            oraciones.append(oracion)
        
        # Removemos StopWords (finándonos en el lema de cada palabra en vez de su texto!)
        # No conviene quitar las StopWords antes de lematizar pues son útiles para ese proceso...
        oraciones = [[palabra for palabra in oracion if palabra[2] not in stopwords] for oracion in oraciones]
        
        # Stemming
        if Stem==True:
            oraciones_aux=[]
            for oracion in oraciones:
                oracion_aux=[]
                for palabra in oracion:
                    p_texto, p_pos, p_lema = palabra
                    # Si Lema es True, se Stemmatiza el lema; si no, se Stemmatiza la palabra original
                    if Lema==True:
                        oracion_aux.append((p_texto, p_pos, p_lema, spanish_stemmer.stem(p_lema)))
                    else:
                        oracion_aux.append((p_texto, p_pos, p_lema, spanish_stemmer.stem(p_texto)))
                oraciones_aux.append(oracion_aux)
            
            oraciones = oraciones_aux
        
        # Finalmente: devolver nuevamente una lista de cadenas como la recibida, pero con el contenido
        # de cada cadena conformado según los parámetros:
        
        Corpus_Procesado = [] #Variable de salida
        
        for doc in oraciones:
            oracion = ''
            for palabra in doc:
                if Stem == True:
                    # Devolver cadena de Stemming
                    oracion = oracion + palabra[3]
                else:
                    if Lema == True:
                        # Devolver cadena de Lemas
                        oracion = oracion + palabra[2]
                    else:
                        # Devolver cadena de palabras originales
                        oracion = oracion + palabra[0]
                
                if POS == True:
                    #Concatenar POS a cada palabra
                    oracion = oracion + '_' + palabra[1].lower()
                
                oracion = oracion + ' '
            
            Corpus_Procesado.append(oracion)
            
        return Corpus_Procesado
    
    
    def Corregir_Documentos(df_textos, columnas, POS=False, Lema=True, Stem=True):
        """
        Recibe un DataFrame con el corpus clasificado a procesar (típicamente con dos columnas:
        Oración y Categoria)
        Devuelve un DataFrame con la misma estructura pero con sus Oraciones filtradas, corregidas,
        y sin redundancias.
        El parámetro columnas es una lista con los nombres de columnas del df_textos a corregir: se
        itera una por una reemplazándola en el df_textos original.
        Los parámetros POS, Lema y Stem se pasan a su vez al PreProcesador y definen su lógica.
        """
        
        for col in columnas:
            df_textos[col] = PreProcesar(list(df_textos[col]), POS, Lema, Stem)
        
        # Sanear el DataFrame eliminando los duplicados y reindexándolo
        df_textos = df_textos.drop_duplicates().reset_index(drop=True)
        
        return df_textos
    
    
    def Generar_Matriz_BOW(df_textos, columna, binario=False, ngram=(1,1)):
        """
        Recibe un DataFrame con el corpus etiquetado a procesar (típicamente dos columnas:
        Oración y Categoria). El segundo parámetro indica el nombre de la columna a modelar, o sea,
        la columna que tiene los documentos a escanear para generar la matriz.
        Devuelve un objeto Vectorizador entrenado con la matriz BOW para las oraciones recibidas, y
        también el dataframe recibido al que se añaden tantas columnas como tokens se hayan identificado
        en el proceso de generación de la matriz, etiquetando cada una con el nombre de dicho token,
        y con sus cantidades contabilizadas en la matriz.
        El parámetro binario establece si cuenta cada ocurrencia del token en el documento, o si sólo
        cuenta uno o cero, sin importar cuántas veces esté presente la misma palabra en el documento.
        El parámetro ngram es una tupla (m,n) que establece el tamaño mínimo y máximo de los ngrams
        a considerar. Para tokenizar sólo con palabras individuales pasar (1,1), para sólo bigramas
        pasar (2,2), para sólo trigramas pasar (3,3), para incluir los tres anteriores pasar (1,3).
        """
        
        # Vectorizar, usando CountVectorizer de sklearn.feature_extraction.text
        from sklearn.feature_extraction.text import CountVectorizer
        vectorizador = CountVectorizer(binary=binario, ngram_range=ngram)
        X = vectorizador.fit_transform(df_textos[columna])
        
        # Generar el DataFrame a devolver
        df_X = pd.DataFrame(X.toarray(), columns=vectorizador.get_feature_names())
        df = df_textos.join(df_X)
        
        return vectorizador, df
    
    
    def Generar_Matriz_Tfidf(df_textos, columna, ngram=(1,1)):
        """
        Recibe un DataFrame con el corpus etiquetado a procesar (típicamente dos columnas:
        Oración y Categoria). El segundo parámetro indica el nombre de la columna a modelar, o sea,
        la columna que tiene los documentos a escanear para generar la matriz.
        Devuelve un objeto Vectorizador entrenado con la matriz Tf*Idf para las oraciones recibidas, y
        también el dataframe recibido al que se añaden tantas columnas como tokens se hayan identificado
        en el proceso de generación de la matriz, etiquetando cada una con el nombre de dicho token,
        y con sus cantidades contabilizadas en la matriz.
        El parámetro ngram es una tupla (m,n) que establece el tamaño mínimo y máximo de los ngrams
        a considerar. Para tokenizar sólo con palabras individuales pasar (1,1), para sólo bigramas
        pasar (2,2), para sólo trigramas pasar (3,3), para incluir los tres anteriores pasar (1,3).
        """
        
        # Vectorizar... Directamente usar aquí el TfidfVectorizer de sklearn en vez del CountVectorizer
        # (Lleva los mismos parámetros y directamente nos devuelve la matriz con los vectores Tf*Idf)
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizador = TfidfVectorizer(ngram_range=ngram)
        X = vectorizador.fit_transform(df_textos[columna])
        
        # Generar el DataFrame a devolver
        df_X = pd.DataFrame(X.toarray(), columns=vectorizador.get_feature_names())
        df = df_textos.join(df_X)
        
        return vectorizador, df
    
    
    def Distancia_Coseno(u, v):
        """
        Recibe dos vectores (de idéntica dimensión, no importa su tamaño)
        Devuelve su distancia coseno: por eso se complementa a 1.
        Interpretación: dos documentos están más próximos mientras más cercano a 0 es su distancia.
        Dos documentos están completamente lejos (no tienen nada en común) si su distancia es 1.
        """
        distancia = 1.0 - (np.dot(u, v) / (np.sqrt(sum(np.square(u))) * np.sqrt(sum(np.square(v)))))
        return distancia
    
    
    #1. Cargar y corregir el corpus
    
    df_textos = pd.read_csv('data_w5.csv', sep=';', encoding = "ANSI")
    #------------------------CORREGIR ARCHIVO------------------------->
    df_textos = df_textos.drop(['Unnamed: 2','Unnamed: 3', 'Unnamed: 4', 'Unnamed: 5'],axis=1)
    df_textos = Corregir_Documentos(df_textos,['oracion'],False,True,True)

    #2. Modelizar los documentos de df_textos
    vectorizador, df_textos = Generar_Matriz_Tfidf(df_textos,'oracion',ngram=(1,3))
    #vectorizador, df_textos = Generar_Matriz_BOW(df_textos,'Oración')

    #3. Separar el corpus en Train/Test
    X = df_textos.drop(['w'],axis=1)
    y = df_textos[['w']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=124)
    X_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)
       
   
    X_new = vectorizador.transform([" ".join([spanish_stemmer.stem(token) for token in new_doc.split() if token not in stopwords])])
    
   
    import pickle
    
    filename= 'w5_modelo_knn'

    modelo_w5 = pickle.load(open(filename,'rb'))
    
    
    
    if (modelo_w5.predict_proba(X_new).max())>0.3:
        return(modelo_w5.predict(X_new)[0])
    
    return('todas')