import spacy
import regex as re



def PreProc(corpus):

    #Minusculizacion.
    corpus = corpus.lower()
    
    #Removiendo puntuacion.
    punct = r'[.,:?!]'
    corpus = re.sub(punct,'', corpus)

    #Tokenizacion con Spacy

    nlp = spacy.load('es_core_news_lg')
    doc = nlp(corpus)
    tokens_str=[str(x) for x in doc]
    
    
    # Cargar lista de palabras y frecuencias

    palabra_frecuencia = {}

    ruta=r"D:\Desktop\ChatBot - PP2\Correcciones de tipeo"
    with open(r"C:\Users\emili\Desktop\CHATBOT FINAL\Palabra_Frecuencia.txt",encoding='utf-8') as f:

        for linea in f:
            (key, val) = linea.split('#')
            palabra_frecuencia[key] = int(val)


    def frecuencia(p):
        try:
            return palabra_frecuencia[p]
        except:
            return 0


    def remove_repeated_characters(tokens,s=10):
   
        repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
        #                             CALL  L  L ES -> "CALLLLES"
        #                             CAL   L  L ES -> "CALLLES"
        #                             CA    L  L ES -> "CALLES"
        #                             CA    L    ES -> "CALES"
        match_substitution = r'\1\2\3'
        
        def replace(old_word):
            if frecuencia(old_word)>s:
                return old_word
            new_word = repeat_pattern.sub(match_substitution, old_word)
            return replace(new_word) if new_word != old_word else new_word
            
        correct_tokens = [replace(word) for word in tokens]
        return correct_tokens

    x = remove_repeated_characters(tokens_str)
    
    WORDS = palabra_frecuencia
    

    def known(words): 
        "The subset of `words` that appear in the dictionary of WORDS."
        return set(w for w in words if w in WORDS)

    def correction(word): 
        "Most probable spelling correction for word."
        return [max(candidates(word), key=frecuencia)]

    def corregir(oracion):
        oracion_corregida=''
        for palabra in x:
            oracion_corregida += correction(palabra)[0] + ' '
        return oracion_corregida

    
    def candidates(word): 
        "Generate possible spelling corrections for word."
        return (known([word]) or 
                known(edits1(word)) or 
                known(edits2(word)) or 
                [word])

    

    

    def edits1(word):
        "All edits that are one edit away from `word`."
        letters    = 'abcdefghijklmnñopqrstuvwxyz'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        #Para "caso":
        #    [('','caso'), -> 'aso
        #     ('c','aso'), -> 'cso
        #     ('ca','so'), -> 'cao
        #     ('cas','o'), -> 'cas
        #     ('caso','')
        #        ]
        
        deletes    = [L + R[1:]               for L, R in splits if R]
        #Para "caso":
        #   ['aso','csa','cao','cas']
        
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        #Para "caso":
        #   ['acso','csao','caos']
        
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        #Para "caso":
        #   ['xaso'...,'cxso'..., 'caxo'..., 'casx'...]
        
        inserts    = [L + c + R               for L, R in splits for c in letters]
        #Para "caso":
        #   ['xcaso'...,'cxaso'..., 'caxso'..., 'casxo'..., 'casox'...]
        
        return set(deletes + transposes + replaces + inserts)

    def edits2(word): 
        "All edits that are two edits away from `word`."
        return (e2 for e1 in edits1(word) for e2 in edits1(e1))

    return(corregir(x))
