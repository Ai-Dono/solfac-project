# -*- coding: utf-8 -*-
"""
Created on Tue Jun 13 15:19:44 2023

@author: axela
"""
#%% Recherche mots clé et Création Url pour Web Scraping
import requests
import json
import wikipedia
from bs4 import BeautifulSoup
from datetime import datetime
import re
from transformers import AutoTokenizer
from transformers import pipeline
import threading

#Chargez les données
tokenizer_path = "../../tokenizer"
model_path = "../pegasus-samsum-model"

TOKEN_HuggingFace = "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"
TOKEN_NY_Times = "7BUBhkD5Ld5zNAsdpvtia7PAAm4OdKdU"
TOKEN_Guardian = "ec200be6-f401-4934-b0bd-9feab9d4c7fe"
# Article de test
fal ="Fargo, ND – Local resident Todd Fox has been detained for “reckless endangerment” and “illegal use of high-powered fire-breathing weaponry” for attacking snow with his flamethrower. Fox reportedly became so fed up with the week-long blowing snow epidemic in his area that he decided to KILL IT WITH FIRE. The neighborhood was treated with quite a show last night as Fox unleashed an inferno upon the mountainous snow palace that was his front yard. Neighbors to his immediate right and left noticed a bright orange cloud and could hear what they thought was “puff the magic dragon spewing mayhem all over hell,” which prompted one of them to notify police. Fox stated that he was simply “fed up with battling the elements” and that he did not possess the willpower necessary to move “four billion tons of white bull shit.” Police say that Fox surrendered his efforts immediately upon their arrival and that his front yard “looked like a hydrogen bomb had gone off.” They think he was just happy to be done with snow removal, even if it did mean a trip to jail."
true = "Ukrainian air defenses shot down 10 of 14 cruise missiles fired by Russia in deadly strikes overnight, the General Staff of the Armed Forces of Ukraine said Tuesday."
true_2 ="César Velez da Silva (born 21 July 1992), commonly known as Cesinha, 22/06/1992 is a Brazilian footballer who currently plays as a forward for Treze, on loan from Perilima."
test ="Un homme de 49 ans a été mis en examen pour « homicide volontaire » après la découverte du corps d’Iris le 27 march. Les enquêteurs recherchent d’autres potentielles victimes, le suspect ayant déjà été condamné pour viol."
ny_times_article = "Federal Reserve officials received an encouraging inflation report on Tuesday as a key price index slowed more than expected in May, news that could give policymakers comfort in pausing interest rate increases at their meeting this week. The Consumer Price Index climbed 4 percent in the year through May, slightly less than the 4.1 percent economists had expected and the slowest pace in more than two years. In April, it had climbed 4.9 percent. While that remains about twice the rate that was normal before the onset of the coronavirus pandemic in 2020, it is down sharply from a peak of about 9 percent last summer. The fresh data offer the latest evidence that the Fed’s push to control rapid price increases is beginning to work. Fed officials have been raising interest rates since March 2022 to make it more expensive to borrow money, in bid to slow consumer demand, tamp down a strong labor market and ultimately cool rapid inflation. They have lifted borrowing costs for 10 meetings in a row, to just above 5 percent, and many officials have suggested in recent weeks that they could soon take a pause to give themselves more time to assess how those adjustments are working. Investors have been betting that Fed officials will leave rates unchanged at their meeting this week, breaking their long streak of increases. But they had also penciled in a small chance that policymakers might lift rates — odds that all but disappeared after Tuesday’s inflation figures. Many investors have also been expecting that Fed officials will restart rate increases in July. After stripping out food and fuel prices, the closely watched measure of “core” prices picked up 5.3 percent in May compared with a year earlier. That was slightly higher than the 5.2 percent economists had expected, but lower than 5.5 percent the previous month. Still, there were lingering signs that inflation has staying power. Fed officials also monitor month-to-month changes in prices, particularly for the core index, to get a sense of the recent trends in inflation. That figure continued to pick up at an unusually quick pace in May. Taken as a whole, the fresh data suggested that while the inflation that has been plaguing consumers and bedeviling the Fed for two years remains stubborn, it is also meaningfully slowing. A cooling economy and a gradually weakening job market could help to further weigh down inflation in the months to come, which could give central bankers confidence that they have lifted borrowing costs enough to bring prices back under control."
usa_today = "WASHINGTON — The Biden and Trump administrations did not sufficiently plan for 'worst-case scenarios' ahead of the U.S. withdrawal of troops from Afghanistan in summer 2021, according to a State Department review released Friday. The report offered strong criticism of the U.S. military withdrawal from Afghanistan, which resulted in the rapid fall of Kabul to the Taliban and the collapse of the Afghan government. The State Department should enhance its crisis planning, clarify its leadership structure during crises, and 'ensure that senior officials hear the broadest possible range of views,' according to the report's recommendations. The ensuing chaos during the American-led evacuation of Kabul — including a terrorist bombing attack — killed more than 150 Afghan citizens and 13 U.S. service members. The withdrawal also resulted in a hasty evacuation of more than 100000 American and Afghan citizens, but also abandoned thousands of other Afghan citizens who supported the U.S. government throughout its 20-year war in Afghanistan. At the time, President Joe Biden and his administration came under intense criticism from Republicans and Democrats alike over his administration's role in the crisis that ensued following the withdrawal. Voters also took notice as Biden's approval ratings fell below 50% for the first time after the messy evacuation from Afghanistan."
#%%  Scrapper des Mois et des Chiffres
def query(payload,API_URL,headers):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

def detect_months_and_numbers(text):
    months = {
        'january': 'January',
        'february': 'February',
        'march': 'March',
        'april': 'April',
        'may': 'May',
        'june': 'June',
        'july': 'July',
        'august': 'August',
        'september': 'September',
        'october': 'October',
        'november': 'November',
        'december': 'December'
    }
    month_regex = r'\b(' + '|'.join(months.keys()) + r')\b'
    matches = re.finditer(month_regex, text, re.IGNORECASE)
    found_months = [(match.group().capitalize(), match.start(), match.end()-1) for match in matches]
    number_matches = re.finditer(r'\b\d+\b', text)
    found_numbers = [(match.group(), match.start(), match.end()-1) for match in number_matches]
    return found_months, found_numbers

def reunir_step(texte, liste_chiffre):
    new_chiffres = []
    i = 0

    while i < len(liste_chiffre):
        if i < len(liste_chiffre) - 1:
            current_number = liste_chiffre[i][0]
            current_start = liste_chiffre[i][1]
            current_end = liste_chiffre[i][2]

            next_number = liste_chiffre[i + 1][0]
            next_start = liste_chiffre[i + 1][1]
            next_end = liste_chiffre[i + 1][2]

            if current_end + 2 == next_start:
                combined_number = current_number + texte[current_end + 1] + next_number
                combined_tuple = (combined_number, current_start, next_end)
                new_chiffres.append(combined_tuple)
                i += 2  
            elif current_end == next_end:
                i += 2  
            elif current_end >= next_start:
                combined_number = current_number + next_number[(current_end - next_start + 1):]
                combined_tuple = (combined_number, current_start, next_end)
                new_chiffres.append(combined_tuple)
                i += 2  
            else:
                new_chiffres.append(liste_chiffre[i])
                i += 1
        else:
            new_chiffres.append(liste_chiffre[i])
            i += 1
    return new_chiffres

def reunir(texte, liste_chiffre):
    new_verif = liste_chiffre
    while new_verif != reunir_step(texte, new_verif) :
        new_verif = reunir_step(texte, new_verif)
    return new_verif

def fusionner_et_trier(liste1, liste2,index,date=False,date_debut = datetime(1900, 1, 1, 0, 0, 0)):
    if liste1 == None:
        liste1 = []
    if liste2 == None:
        liste2 = []
    liste_fusionnee = liste1 + liste2
    if date == True and date_debut == datetime(1900, 1, 1, 0, 0, 0):
        liste_triee = sorted(liste_fusionnee, key=lambda x: datetime.strptime(x[index], "%m/%d/%Y"),reverse=True)
    elif date == True and date_debut != datetime(1900, 1, 1, 0, 0, 0) :
        liste_triee = sorted(liste_fusionnee, key=lambda x: datetime.strptime(x[index], "%m/%d/%Y"),reverse=True)
        liste_triee = [elem for elem in liste_triee if datetime.strptime(elem[index], "%m/%d/%Y") >= date_debut]
    else:
        liste_triee = sorted(liste_fusionnee, key=lambda x: x[index])
    return liste_triee
def liste_date(liste):
    new_liste = set()
    for i in liste:
        if '.' not in i[0] and len(i[0]) > 2 and not i[0].isdigit():
            new_liste.add(i[0])
        elif i[0].isdigit() and datetime.now().year >= int(i[0]) >= 1900:
            new_liste.add(i[0])
    return new_liste
def best_date(liste):
    new_liste = []
    liste_max = [0]
    for i in liste:
         liste_max.append(i.count(' '))
         liste_max.append(i.count('/'))
    maximum = max(liste_max)
    for i in liste:
        if i.count(' ') == maximum or i.count('/') == maximum :
         new_liste.append(i)
    return new_liste
def plus_ancienne(liste):
    result = datetime(1900, 1, 1, 0, 0, 0)
    parsed_dates = []
    for date in liste :
        if date.count(' ') == 2 or date.count('/') == 2:
                try:
                    parsed_date = datetime.strptime(date, "%d %B %Y")
                except ValueError:
                    parsed_date = datetime.strptime(date, "%d/%m/%Y")
                parsed_dates.append(parsed_date)
        elif date.count(' ') == 1 or date.count('/') == 1:
                try:
                    parsed_date = datetime.strptime(date, "%B %Y")
                except ValueError:
                    try:
                        parsed_date = datetime.strptime(date+" 2000", "%d %B %Y")
                    except ValueError:
                        parsed_date = datetime.strptime(date, "%m/%Y")
                parsed_dates.append(parsed_date)
        elif date.count(' ') == 0 or date.count('/') == 0:
                try:
                    parsed_date = datetime.strptime(date+" 2000", "%B %Y")
                    parsed_date.year = 2000
                except ValueError:
                    parsed_date = datetime.strptime(date, "%Y")
                parsed_dates.append(parsed_date)
        try :
           result = min(parsed_dates)
        except ValueError: 
            result = datetime(1900, 1, 1, 0, 0, 0)
    return result
def date_dans_texte(texte):
    mois, chiffres = detect_months_and_numbers(texte)
    ensemble_date = fusionner_et_trier(chiffres,mois,1)
    new_ensemble_date = plus_ancienne(best_date(liste_date(reunir(texte,ensemble_date))))
    return new_ensemble_date

def resumer(texte):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    gen_kwargs = {"length_penalty": 0.8, "num_beams":8, "max_length": 128}
    pipe = pipeline("summarization", model=model_path,tokenizer=tokenizer)
    res = pipe(texte, **gen_kwargs)[0]["summary_text"]
    return res

def process_text(text):
    sentences = text.split(".")
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]
    last_sentence = sentences[-1]
    if last_sentence[-1] not in [".", "?", "!"]:
        sentences = sentences[:-1]
    processed_sentences = [sentence.capitalize() for sentence in sentences]
    processed_text = ". ".join(processed_sentences) + "."
    return processed_text

#%% Mots clés
def key_word(texte):
    TOKEN_HuggingFace = "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"
    API_URL = "https://api-inference.huggingface.co/models/yanekyuk/bert-uncased-keyword-extractor"
    headers = {"Authorization": f"Bearer {TOKEN_HuggingFace}"}
    reponse_false = True
    while reponse_false :	
        output = query({
        	"inputs": texte,
        },API_URL,headers)
        if "error" in output  :
            reponse_false = True
        else:
            reponse_false = False
    
    keyword = []
    
    for i in range(len(output)):
        if output[i]["score"] >= 0.90 :
            keyword.append(output[i]["word"])
    return list(set(keyword)),date_dans_texte(texte)

def printlistdict(info,n=0):
    if n<=0 :
        for i in range(len(info)):
            print("Title : " + info[i]["title"]) 
            print("Resumme : \n" + info[i]["content"])
            print("-------------------------------------------------------")
            print("URL : " + info[i]["url"]) 
            print("Source : " + info[i]["source"])
            if "date" in info[i].keys():
                print("Publication date : " + info[i]["date"]) 
            print("-------------------------------------------------------")
            if "score" in info[i].keys():
                print("Score de vraisemblance : {:.2f}%".format(info[i]["score"]*100)) 
                print("-------------------------------------------------------")
    else :
        for i in range(n):
            print("Title : " + info[i]["title"]) 
            print("Resumme : \n" + info[i]["content"])
            print("-------------------------------------------------------")
            print("URL : " + info[i]["url"]) 
            print("Source : " + info[i]["source"])
            if "date" in info[i].keys():
                print("Publication date : " + info[i]["date"]) 
            print("-------------------------------------------------------")
            if "score" in info[i].keys():
                print("Score de vraisemblance : {:.2f}%".format(info[i]["score"]*100)) 
                print("-------------------------------------------------------")


#%% Similarity
def compute_similarity_score(word1, word2):
    set1 = set(word1.lower())
    set2 = set(word2.lower())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    similarity_score = len(intersection) / len(union)
    return similarity_score

#%% Wiki
def research_wiki(keyword):
    wikipedia.set_lang("en")
    res= []
    keyword = sorted(keyword, key=lambda x: len(x), reverse=True)
    termes_finaux = []
    
    for terme in keyword:
        if all(terme not in t and t not in terme for t in termes_finaux):
            termes_finaux.append(terme)
    for i in range(len(termes_finaux)) : 
        title_wiki = []
        content_wiki = []
        url_wiki = []
        term_wiki = wikipedia.search(termes_finaux[i],results = 10, suggestion = True)[0]
        n= 0
        while n !=10 :
            try :
                title_wiki.append(wikipedia.page(term_wiki[n]).title)
                content_wiki.append(wikipedia.summary(term_wiki[n],sentences = 4))
                url_wiki.append(wikipedia.page(term_wiki[n]).url)
            except Exception:
               n = n 
            n += 1
        score = []
        for name in title_wiki :
            score.append(compute_similarity_score(termes_finaux[i], name))
        index = score.index(max(score))
        res.append({"title":title_wiki[index],"content":content_wiki[index],"url":url_wiki[index],"source":"Wikipédia"})
    res_fin = []
    for i in res : 
        if i not in res_fin: 
            res_fin.append(i) 
    return  sorted(res_fin, key=lambda x: x["title"])



   
#%% New York Times

def research_NY_Times(keyword):
    TOKEN_NY_Times = "7BUBhkD5Ld5zNAsdpvtia7PAAm4OdKdU"
    url_ny_times = "https://api.nytimes.com/svc/search/v2/articlesearch.json?q="
    for i in range(len(keyword)) :
        url_ny_times += "+" + keyword[i].replace(" ", "+")
    try:
        url_ny_times += "&api-key="+TOKEN_NY_Times
        req = requests.get(url_ny_times)
        soup = BeautifulSoup(req.text,'html.parser').prettify()
    except Exception as e:
        print(e)
    res = []
    data_ny_times = json.loads(soup)
    data_title_ny_times = []
    data_paragraph_ny_times = []
    data_url_ny_times = []
    data_source_ny_times = []
    data_publication_ny_times = []
    for i in range(len(data_ny_times['response']['docs'])) :
        
        if data_ny_times['response']['docs'][i]['document_type'] == "article" :
            data_title_ny_times.append(data_ny_times['response']['docs'][i]['headline']["main"])
            data_paragraph_ny_times.append(data_ny_times['response']['docs'][i]['lead_paragraph'])
            data_url_ny_times.append(data_ny_times['response']['docs'][i]['web_url'])
            data_source_ny_times.append(data_ny_times['response']['docs'][i]['source'])
            data_publication_ny_times.append(datetime.strptime(data_ny_times['response']['docs'][i]['pub_date'], "%Y-%m-%dT%H:%M:%S%z").strftime("%m/%d/%Y"))
    for i in range(len(data_title_ny_times)):
        res.append({"title":data_title_ny_times[i],"content":data_paragraph_ny_times[i],"url":data_url_ny_times[i],"source":data_source_ny_times[i],"date":data_publication_ny_times[i]})
    res_fin = []
    for i in res : 
        if i not in res_fin: 
            res_fin.append(i) 
    return sorted(res_fin, key=lambda x: datetime.strptime(x["date"], "%m/%d/%Y"),reverse=True)


    
#%% The Guardian 
def research_Guardian(keyword,date):
    url_guardian = "https://content.guardianapis.com/search?q="
    
    for i in range(len(keyword)) :
        url_guardian += "+" + keyword[i].replace(" ", "%20AND%20")
    try:
        url_guardian += "&from-date="+date.strftime("%Y-%m-%d")+"&api-key="+TOKEN_Guardian
        req = requests.get(url_guardian)
        soup = BeautifulSoup(req.text,'html.parser').prettify()
    except Exception as e:
        print(e)
    
    data_guardian = json.loads(soup)
    res = []
    data_paragraph_guardian = []
    data_url_guardian = []
    data_title_guardian = []
    data_publication_guardian = []
    for i in range(len(data_guardian['response']['results'])) :
        if data_guardian['response']['results'][i]['type'] == "article" :
            data_title_guardian.append(data_guardian['response']['results'][i]['webTitle'])
            data_url_guardian.append(data_guardian['response']['results'][i]['webUrl'])
            data_publication_guardian.append(datetime.strptime(data_guardian['response']['results'][i]['webPublicationDate'], "%Y-%m-%dT%H:%M:%SZ").strftime("%m/%d/%Y"))

            try:
                url_guardian = data_guardian['response']['results'][i]['webUrl']
                req = requests.get(url_guardian)
                soup = BeautifulSoup(req.text,'html.parser').find_all(lambda tag: tag.name == 'p' and tag.get('class') and any(cls.startswith('dcr') for cls in tag.get('class')))
            except Exception as e:
                print(e)
            resume= ""
            for x in soup:
                resume += re.sub('<.*?>', '', str(x))
            data_paragraph_guardian.append(process_text(resumer(resume)))         
    for i in range(len(data_paragraph_guardian)):
        res.append({"title":data_title_guardian[i],"content":data_paragraph_guardian[i],"url":data_url_guardian[i],"source":"The Guardian","date":data_publication_guardian[i]})
    res_fin = []
    for i in res : 
        if i not in res_fin: 
            res_fin.append(i) 
    return  sorted(res_fin, key=lambda x: datetime.strptime(x["date"], "%m/%d/%Y"),reverse=True)


#%% Noter la vraisemblance 
import torch
from transformers import BertTokenizer, BertModel

# Load the pre-trained BERT model and 
device = "cuda" if torch.cuda.is_available() else "cpu"
model = BertModel.from_pretrained("bert-base-uncased")#.to(device)
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
def scoring(article,list_article):
    TOKEN_HuggingFace = "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"
    API_URL = "https://api-inference.huggingface.co/models/google/pegasus-cnn_dailymail"
    headers = {"Authorization": f"Bearer {TOKEN_HuggingFace}"}	
    text = str(article)
    output = query({"inputs": text,},API_URL,headers)
     
    # Example sentences
    sentence1 = output[0]['summary_text']
    for i in list_article :
        sentence2 = i["content"]
        
        # Tokenize and encode the sentences
        tokens = tokenizer([sentence1, sentence2], padding=True, truncation=True, return_tensors="pt")
        input_ids = tokens.input_ids
        attention_mask = tokens.attention_mask
        
        # Forward pass through the model
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        sentence_embeddings = outputs.last_hidden_state
        
        # Calculate the similarity score for each pair of sentences
        similarity_scores = torch.nn.functional.cosine_similarity(sentence_embeddings[0], sentence_embeddings[1], dim=1)
        result = similarity_scores.mean().item()
        i["score"] = result
    
    return list_article

#%% Fonction Ultime

class ResultThread(threading.Thread):
    def __init__(self, target):
        super(ResultThread, self).__init__(target=target)
        self.result = None

def ultimate_result(texte):
    keyword = key_word(texte)

    # Création des threads pour chaque recherche
    wiki_thread = ResultThread(target=lambda: setattr(wiki_thread, 'result', research_wiki(keyword[0])))
    guardian_thread = ResultThread(target=lambda: setattr(guardian_thread, 'result', scoring(texte, research_Guardian(keyword[0], keyword[1]))))
    nytimes_thread = ResultThread(target=lambda: setattr(nytimes_thread, 'result', scoring(texte, research_NY_Times(keyword[0]))))

    # Lancement des threads
    wiki_thread.start()
    guardian_thread.start()
    nytimes_thread.start()

    # Attendre la fin de tous les threads
    wiki_thread.join()
    guardian_thread.join()
    nytimes_thread.join()

    # Récupérer les résultats
    wiki_result = wiki_thread.result
    guardian_result = guardian_thread.result
    nytimes_result = nytimes_thread.result

    liste_article = fusionner_et_trier(guardian_result, nytimes_result, "score")
    return [wiki_result, liste_article]

#%% Test
import time

# # Enregistrer le temps de départ
# temps_debut = time.time()

# article_choisie = usa_today
# keyword = key_word(article_choisie)
# print("L'article choisie :\n")
# print(article_choisie)
# print("\nLes mot clés sont :")
# print(keyword[0])
# print("\nLa date la plus ancienne de l'article :")
# print(keyword[1])

# print("\nLes articles permettant de comprendres les mots clés :\n")
# #Recherche des articles Wikipedia des mots clés
# wiki = research_wiki(keyword[0])
# #Print des articles   
# printlistdict(wiki)  

# print("\nLes articles venant de l'API du New York Time liez aux mots clés :\n")
# #Recherche des articles du New York Times liez aux mots clés
# NY_time = research_NY_Times(keyword[0])
# #Print des articles   
# printlistdict(NY_time)

# print("\nLes articles venant de l'API du Guardian liez aux mots clés :\n")
# #Recherche des articles du Guardian liez aux mots clés
# #Prend du temps car il céer les résummez des articles
# Guardian = research_Guardian(keyword[0],keyword[1])   
# #Print des articles 
# printlistdict(Guardian)

# print("\nLes articles trouvées :\n")
# #Fusion de l'ensemble des articles triez par date en supprimant tous ceux datant d'aprés la date la plus ancienne
# liste_article = fusionner_et_trier(Guardian,NY_time,"date",True,keyword[1])
# #Print les 5 articles les plus récent de l'ensemble des articles
# printlistdict(liste_article,n=5)

# print("\nLes articles trouvées et noté :\n")
# #Notation de la liste d'article
# liste_article = scoring(article_choisie, liste_article)
# #Print les 5 articles les plus récent de l'ensemble des articles
# printlistdict(liste_article,n=5)

# print("\nArticle avec le meilleur score :\n")
# #Notation de la liste d'article
# best_article = [max(liste_article, key=lambda x: x['score'])] 
# printlistdict(best_article)

# temps_fin = time.time()

# # Calculer la durée totale
# duree = temps_fin - temps_debut

# # Afficher le temps de calcul
# print("Temps de calcul:", duree, "secondes")
#%%
test=[[{'title': 'COVID-19 pandemic cases', 'content': "The article contains the number of cases of coronavirus disease 2019 (COVID-19) reported by each country, territory, and subnational area to the World Health Organization (WHO) and published in WHO reports, tables, and spreadsheets. As of 3 July 2023, 767,517,959 cases have been stated by government agencies from around the world to be confirmed. For more international statistics in table and map form, see COVID-19 pandemic by country and territory.\n108 countries and territories have more confirmed cases than the People's Republic of China, the country where the outbreak began.", 'url': 'https://en.wikipedia.org/wiki/COVID-19_pandemic_cases', 'source': 'Wikipédia'}, {'title': 'Consumer price index', 'content': 'A consumer price index (CPI) is a price index, the price of a weighted average market basket of consumer goods and services purchased by households. Changes in measured CPI track changes in prices over time.\n\n\n== Overview ==\nA CPI is a statistical estimate constructed using the prices of a sample of representative items whose prices are collected periodically. Sub-indices and sub-sub-indices can be computed for different categories and sub-categories of goods and services, being combined to produce the overall index with weights reflecting their shares in the total of the consumer expenditures covered by the index.', 'url': 'https://en.wikipedia.org/wiki/Consumer_price_index', 'source': 'Wikipédia'}, {'title': 'Federal Reserve', 'content': "The Federal Reserve System (often shortened to the Federal Reserve, or simply the Fed) is the central banking system of the United States. It was created on December 23, 1913, with the enactment of the Federal Reserve Act, after a series of financial panics (particularly the panic of 1907) led to the desire for central control of the monetary system in order to alleviate financial crises. Over the years, events such as the Great Depression in the 1930s and the Great Recession during the 2000s have led to the expansion of the roles and responsibilities of the Federal Reserve System.Congress established three key objectives for monetary policy in the Federal Reserve Act: maximizing employment, stabilizing prices, and moderating long-term interest rates. The first two objectives are sometimes referred to as the Federal Reserve's dual mandate.", 'url': 'https://en.wikipedia.org/wiki/Federal_Reserve', 'source': 'Wikipédia'}, {'title': 'Inflationism', 'content': 'Inflationism is a heterodox economic, fiscal, or monetary policy, that predicts that a substantial level of inflation is harmless, desirable or even advantageous. Similarly, inflationist economists advocate for an inflationist policy.\nMainstream economics holds that inflation is a necessary evil, and advocates a low, stable level of inflation, and thus is largely opposed to inflationist policies – some inflation is necessary, but inflation beyond a low level is not desirable. However, deflation is often seen as a worse or equal danger, particularly within Keynesian economics, as well as Monetarist economics and in the theory of debt deflation.', 'url': 'https://en.wikipedia.org/wiki/Inflationism', 'source': 'Wikipédia'}], [{'title': 'From CPI to stagflation: how the UK tracks price rises and what key inflation terms mean', 'content': 'On its website the bank of england has a tool allowing users to find out how prices have changed ever since king john was on the english throne back in 1209. By its calculations, goods and services costing £10 six years before magna carta was sealed would today cost £16,427. 96.', 'url': 'https://www.theguardian.com/business/2023/may/23/cpi-stagflation-uk-price-rises-key-inflation-terms', 'source': 'The Guardian', 'date': '05/23/2023', 'score': 0.2448180466890335}, {'title': 'Fed’s Favorite Inflation Gauge Remains High, but Gains Slowed in May', 'content': 'Inflation climbed in May at the fastest pace since 2008, as businesses reopening from pandemic shutdowns and strong demand continued to push prices higher, fueling anxiety among some economists and debate in Washington.', 'url': 'https://www.nytimes.com/2021/06/25/business/inflation-federal-reserve.html', 'source': 'The New York Times', 'date': '06/25/2021', 'score': 0.3007901608943939}, {'title': 'Australia’s softening inflation unlikely to spell an end to interest rate hikes', 'content': 'Anz expects that electricity prices jumped 13% in the final three months of the year alone, which would be the largest quarterly increase since the introduction of carbon pricing in 2012.', 'url': 'https://www.theguardian.com/australia-news/2023/jan/25/australias-softening-inflation-unlikely-to-spell-an-end-to-interest-rate-hikes', 'source': 'The Guardian', 'date': '01/24/2023', 'score': 0.3049928545951843}, {'title': 'Inflation retreated in May to 5.6%, easing fears RBA will again raise interest rates ', 'content': 'Australia’s monthly inflation rate retreated in may, easing fears the reserve bank will hoist its key interest rate again at next tuesday’s board meeting. The headline consumer price index increase last month was 5. 6%, the lowest since april 2022, the australian bureau of statistics said on wednesday. Economists had expected the measure to drop from april’s 6. 8% level to 6. 1%. Housing costs were among the biggest contributors to the monthly cpi numbers, rising 8. 4%, down from 8.', 'url': 'https://www.theguardian.com/australia-news/2023/jun/28/inflation-retreated-in-may-to-56-easing-fears-rba-will-again-raise-interest-rates', 'source': 'The Guardian', 'date': '06/28/2023', 'score': 0.3086128532886505}, {'title': 'Consumer prices popped again in December, casting a shadow over the economy.', 'content': 'Inflation climbed to its highest level in 40 years at the end of 2021, a troubling development for President Biden and economic policymakers as rapid price gains erode consumer confidence and cast a shadow of uncertainty over the economy’s future.', 'url': 'https://www.nytimes.com/2022/01/12/business/economy/cpi-inflation-december-2021.html', 'source': 'The New York Times', 'date': '01/12/2022', 'score': 0.3129394054412842}, {'title': 'Prices Pop Again, and Fed and White House Seek to Ease Inflation Fears', 'content': 'A key measure of inflation spiked in June, climbing at the fastest pace in 13 years as prices for used cars, hotel stays and restaurant meals surged while the economy reopens.', 'url': 'https://www.nytimes.com/2021/07/13/business/economy/consumer-price-index-june-2021.html', 'source': 'The New York Times', 'date': '07/13/2021', 'score': 0.31395524740219116}, {'title': 'Prices jumped 3.6 percent in April, the fastest pace in 13 years.', 'content': 'Prices are climbing at the fastest pace since 2008, a key index released on Friday showed, an increase that is sure to keep inflation central to economic and political debates.', 'url': 'https://www.nytimes.com/2021/05/28/business/inflation-consumer-prices.html', 'source': 'The New York Times', 'date': '05/28/2021', 'score': 0.3155362904071808}, {'title': 'UK inflation falls by less than expected as food prices soar by 19.1%', 'content': 'The odds of the bank of england raising its base interest rate next month jumped on the news, with markets pricing in a 97% chance of an increase to 4. 5% on 11 may, and indicating it could hit 5% by the autumn.', 'url': 'https://www.theguardian.com/business/2023/apr/19/uk-inflation-falls-consumer-prices-index', 'source': 'The Guardian', 'date': '04/19/2023', 'score': 0.31730568408966064}, {'title': 'Consumer Prices Keep Climbing as Fed and White House Await a Cool-Down', 'content': 'Consumer prices rose at a rapid clip in July as the reopening of the economy kept inflation elevated at levels that are making officials at the Federal Reserve squirm while posing a political problem for the Biden White House.', 'url': 'https://www.nytimes.com/2021/08/11/business/economy/july-2021-consumer-price-inflation.html', 'source': 'The New York Times', 'date': '08/11/2021', 'score': 0.31832072138786316}, {'title': 'Fast Price and Wage Growth Keeps Fed on Track for Big Interest Rate Increase', 'content': 'The price index that the Federal Reserve watches most closely climbed 6.6 percent in the year through March, the fastest pace of inflation since 1982 and the latest reminder of the painfully rapid price increases plaguing consumers and challenging policymakers.', 'url': 'https://www.nytimes.com/2022/04/29/business/economy/pce-inflation-march-2022.html', 'source': 'The New York Times', 'date': '04/29/2022', 'score': 0.3183344900608063}, {'title': 'Shop price inflation easing, say top UK retailers before key meeting with MPs', 'content': 'The british retail consortium (brc) said annual inflation in overall shop prices eased to 8. 4% in june, down from 9% in may, as retailers cut the price of many staples including milk, cheese and eggs. Clothing and electrical goods prices also fell ahead of the summer holidays. The retail industry is facing growing pressure over the soaring cost of living, after official figures showed the uk’s annual inflation rate remained unchanged in april at 8. 7%.', 'url': 'https://www.theguardian.com/business/2023/jun/27/shop-price-inflation-easing-say-top-uk-retailers-before-key-meeting-with-mps', 'source': 'The Guardian', 'date': '06/26/2023', 'score': 0.3220959007740021}, {'title': 'Consumers are on the lookout for higher prices, a Federal Reserve survey shows.', 'content': 'Expectations for long-term inflation rose slightly in July, the Federal Reserve Bank of New York said on Monday, and consumers continued to foresee rapid price gains in the near term as the economy reopens from pandemic-related lockdowns.', 'url': 'https://www.nytimes.com/2021/08/09/business/consumer-prices-inflation.html', 'source': 'The New York Times', 'date': '08/09/2021', 'score': 0.3265485465526581}, {'title': 'Afternoon Update: inflation drops more than expected; William Tyrrell’s foster mother speaks; and a cleaner destroys 25 years of research', 'content': 'Inflation fell to 5. 6% in may – lower than economists’ expectations of 6. 1% and potentially easing fears of a rate rise next month. But monthly inflation data can be volatile and a closer look reveals varying price movements. And if you’re surprised by your higher insurance premium this year, it’s because insurers have hiked them up a record 14. 2% – 2. 5 times the inflation rate.', 'url': 'https://www.theguardian.com/australia-news/2023/jun/28/afternoon-update-inflation-drops-more-than-expected-william-tyrrells-foster-mother-speaks-and-a-cleaner-destroys-25-years-of-research', 'source': 'The Guardian', 'date': '06/28/2023', 'score': 0.3298102617263794}, {'title': 'US inflation at 5%, the lowest it has been since 2021', 'content': 'In february, the annual inflation figure stood at 6%, already a steep decline from its peak of 9. 1% in june. But core inflation, which does not include volatile energy and food prices, has remained steady – a sign that the slowing pace could be attributed to comparisons against soaring gas prices a year ago, near the beginning of russia’s invasion of ukraine. In march, the fed increased rates by a quarter point to a range of 4.', 'url': 'https://www.theguardian.com/business/2023/apr/12/cpi-inflation-rate-march-prices-fed', 'source': 'The Guardian', 'date': '04/12/2023', 'score': 0.3434402048587799}, {'title': 'Prices climbed more slowly in August, welcome news for the Fed.', 'content': 'A recent run-up in consumer prices cooled slightly in August, signaling that although inflation is higher than normal, the White House and Federal Reserve may be beginning to see the slowdown in price gains they have been hoping for.', 'url': 'https://www.nytimes.com/2021/09/14/business/consumer-price-index-august-2021.html', 'source': 'The New York Times', 'date': '09/14/2021', 'score': 0.34658464789390564}, {'title': 'Prices Jumped 5% in May From Year Earlier, Stoking Debate in Washington', 'content': 'Consumer prices rose in May at the fastest rate since 2008, a bigger jump than economists had expected and one that is sure to keep inflation at the center of political and economic debate in Washington.', 'url': 'https://www.nytimes.com/2021/06/10/business/consumer-price-index-may-2021.html', 'source': 'The New York Times', 'date': '06/10/2021', 'score': 0.3508423864841461}, {'title': 'The Fed’s Favorite Price Index Rose 4 Percent. What Comes Next?', 'content': 'The Federal Reserve’s preferred measure of inflation climbed by 4 percent in June compared with a year earlier, as a rebounding economy and strong demand for goods and services helped to push prices higher.', 'url': 'https://www.nytimes.com/2021/07/30/business/economy/pce-inflation-federal-reserve.html', 'source': 'The New York Times', 'date': '07/30/2021', 'score': 0.36975812911987305}]]


from flask import Flask, redirect,render_template, url_for,request
app = Flask(__name__)
    
common={}
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        textarea_value = request.form['my_textarea']
        results=ultimate_result(textarea_value)
        titlelist=[]
        contentlist=[]
        urlist=[]
        articaffich=[]
        for i in range(len(results[0])):
            titlelist.append((results[0][i]['title']))
        for i in range(len(results[0])):
            contentlist.append((results[0][i]['content']))
        for i in range(len(results[0])):
            urlist.append((results[0][i]['url']))
        if len(results[1])>5:
            for i in results[1][-5:]:
                articaffich.append([i['url'],i['content']])
                
        common={
                'title': titlelist,
                'content': contentlist,
                'url': urlist,
                'articaffich':articaffich

                }
            
                    # Faites quelque chose avec textarea_value, par exemple le stocker dans une base de données ou l'utiliser dans votre logique métier
        return render_template('result.html',common=common)
    
    return render_template('home.html')

@app.route('/result')
def result():
    return render_template('result.html',common=common)
@app.route('/change_page')
def change_page():
    return redirect(url_for('result'))


if __name__ == "__main__":
    print("running py app")
    app.run(host="127.0.0.1", port=5000, debug=True)
            

