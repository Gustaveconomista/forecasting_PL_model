# %% [markdown]
# ## Importe dos Pacotes

# %% [markdown]
# Começamos instalando as bibliotecas e importando os pacotes necessários.

# %% [markdown]
# `!pip3 install snscrape`
# 
# `!pip install tabulate`
# 
# `!pip install wordcloud`
# 
# `!pip install selenium`

# %%
import pandas as pd
import numpy as np
import requests
import time
from bs4 import BeautifulSoup
from selenium import webdriver
import mysql.connector as mdb
from PIL import Image
import nltk
import os
import re
import zipfile
import matplotlib.pyplot as plt
import networkx as nx
import snscrape.modules.twitter as sntwitter
from nltk.sentiment import SentimentIntensityAnalyzer as sia
import seaborn as sns
import itertools
from wordcloud import WordCloud
from tabulate import tabulate
import unicodedata
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from math import pi
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
nltk.download('vader_lexicon')
nltk.download('stopwords')
nltk.download('punkt')

# %% [markdown]
# ## Coleta/Raspagem de Dados

# %% [markdown]
# Criando um conector do SQL para armazenar nossos dados de maneira profissional. 

# %%
con = mdb.connect(user='root', password ='*********')
cursor = con.cursor()

# %% [markdown]
# Num primeiro passo, definimos uma lista com o nome dos clubes do campeonato e seus respectivos apelidos mais utilizados pelos torcedores (termos chave).

# %%
pl = ['(Man City OR Cityzens)', '(Liverpool OR Reds)',
               '(Chelsea OR Blues)', '(Tottenham OR Lilywhites OR Spurs)', '(Arsenal OR Gunners)', 
                'Man United', '(West Ham OR Hammers OR Irons)',
               '(Leicester OR Foxes)', '(Brighton OR Seagulls)', '(Wolverhampton OR Wolves)',
                '(Newcastle OR Magpies OR Toon OR Geordies)', '(Crystal Palace OR Palace OR Eagles)',
               '(Brentford OR Bees)', '(Aston Villa OR Villa OR Villans OR Lions)', '(Southampton OR Saints)',
               '(Everton OR Toffees OR Toffeemen)', '(Leeds OR Whites OR Peacocks)', '(Burnley OR Clarets)',
               '(Watford OR Hornets)', '(Norwich OR Canaries OR Yellows)']
pl

# %% [markdown]
# Criamos, em seguida, uma função lambda que mapeia o nome dos clubes por seus respectivos apelidos, criando, assim, uma coluna onde cada linha referencia o nome de um clube associado ao twiiter sobre ele.

# %%
class MissingDict1(dict):
    __missing__ = lambda self, key: key

map_values1 = {"(Man City OR Cityzens)": "Manchester City",
              "(Liverpool OR Reds)": "Liverpool",
              "(Chelsea OR Blues)": "Chelsea",
              "(Arsenal OR Gunners)": "Arsenal",
              "(Leicester OR Foxes)": "Leicester",
              "(Brighton OR Seagulls)": "Brighton",
              "Man United": "Manchester Utd",
              "(Newcastle OR Magpies OR Toon OR Geordies)": "Newcastle Utd",
              "(Tottenham OR Lilywhites OR Spurs)": "Tottenham",
              "(West Ham OR Hammers OR Irons)": "West Ham",
              "(Wolverhampton OR Wolves)": "Wolves",
              "(Crystal Palace OR Palace OR Eagles)": "Crystal Palace",
              "(Brentford OR Bees)": "Brentford",
              "(Aston Villa OR Villa OR Villans OR Lions)": "Aston Villa",
              "(Southampton OR Saints)": "Southampton",
              "(Everton OR Toffees OR Toffeemen)": "Everton",
              "(Leeds OR Whites OR Peacocks)": "Leeds",
              "(Burnley OR Clarets)": "Burnley",
              "(Watford OR Hornets)": "Watford",
              "(Norwich OR Canaries OR Yellows)": "Norwich"}
mapping1 = MissingDict1(**map_values1)

# %% [markdown]
# Utilizando o modulo do twitter da blibioteca snscrape, coletamos os tweets relativos aos clubes da Premier League (PL).
# Para isso, aplicamos uma função que formata as entradas de texto do df de modo compatível com a syntax do SQL, uma outra para listar todas as semanas da temporada e uma para coletar dos dados.

# %%
def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except (TypeError, NameError):  
        pass
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore')
    text = text.decode("utf-8")
    return str(text)

def text_to_sql(text):
    text = strip_accents(text.lower())
    text = re.sub('[^@$!?&.#0-9a-zA-Z_-]'," ", text)
    text= text.lstrip()
    return text

# %%
def all_weeks(date):
    return pd.date_range(start=str(date),
                        end=str('2022-05-15'),
                        freq="W").strftime("""
                                           %Y-%m-%d""").tolist()

# %%
def weekly_data(week, keywords):
    week = pd.to_datetime(week)
    until = week.strftime('%Y-%m-%d')
    since = (week - pd.to_timedelta(1, 'W')).strftime('%Y-%m-%d')
    query = str(keywords) + ' AND (premier league OR pl) lang:en min_faves:0 since:{} until:{}'.format(since, until)
    tweets_list = []
    for tweet in itertools.islice(sntwitter.TwitterSearchScraper(query).get_items(), 0,5000,None):
        fields = {}
        date = str(tweet.date)     
        text = str(tweet.content)
        text = strip_accents(text)
        text = text_to_sql(text)
        username = str(tweet.user.username)
        favorites = str(tweet.likeCount)
        retweets = str(tweet.retweetCount)
        keyterm = str(keywords)
        fields['datetime'] = date
        fields['username'] = username
        fields['text'] = text
        fields['favorites'] = favorites
        fields['retweets'] = retweets
        fields['keyterm'] = keyterm
        tweets_list.append(fields)
    data = pd.DataFrame(tweets_list)
    return data

# %% [markdown]
# Aqui, definimos uma função da qual utilizamos a library de Machine Learning SentimentIntensityAnalyzer para identificar o sentimento que tais tweets sobre os clubes expressam, se são positivos, negativos ou neutros.

# %%
def sentiment(tweet: str) -> bool:      
    if sia().polarity_scores(tweet)["compound"]>0:
        return 'positive'
    elif sia().polarity_scores(tweet)["compound"]<0:
        return 'negative'
    elif sia().polarity_scores(tweet)["compound"]==0:
        return 'neutral'

# %% [markdown]
# Aqui temos o loop para realizar a coleta com o devido corte temporal de interesse, a aplicação da função de sentimento e algumas manipulações de nosso df, para no final, salva-lo em um arquivo csv.

# %%
x = all_weeks('2021-08-07')
f = []
for i in range(0, len(pl)):
    for j in range(0, len(x)):
        w = weekly_data(x[j], pl[i])
        f.append(w)
df1 = pd.concat(f)
df1['sentiment']=df1.loc[:, 'text'].apply( lambda x: sentiment(x))
df1['datetime'] = pd.to_datetime(df1.datetime, format='%Y-%m-%d')
df1['date'] = df1['datetime'].dt.date
df1["team"] = df1["keyterm"].map(mapping1)
df1.to_csv('C:/Users/guhhh/OneDrive/Área de Trabalho/Workspace/modeloPL/tts_pl_teams.csv')

# %% [markdown]
# Criaremos agora nossa database no SQL, de modo a armazenar todos os dados deste projeto nela.

# %%
query_db = "CREATE DATABASE modelopl"
cursor.execute(query_db)

# %%
# Selecionando nossa database
query_sdb = "USE modelopl"
cursor.execute(query_sdb)

# %% [markdown]
# Para automatizar o processo, criamos abaixo uma função que escreve nossa query de criação da tabela na database do SQL.

# %%
def create_tb(n, lista):
    l = ["CREATE TABLE " + str(n) + " ("]
    for i in range(0, (len(lista)-1)):
        q = lista[i] + ' TEXT NOT NULL,'
        l.append(q)
    l.append(lista[-1] + " TEXT NOT NULL)")
    query = ' '.join([str(item) for item in l])
    return query

# %%
cols = df1.columns.tolist()
cols

# %%
query_tb = create_tb('ttspl',cols)
query_tb

# %%
# Criando agora a tabela em nosso servidor SQL, dentro da database que acabamos de gerar acima, de modo a armazenar o df dos tts
cursor.execute(query_tb)

# %%
# Armazenando o df dos tts
for i in range(0, len(df1)):
    query_insert = ("INSERT INTO ttspl (datetime, username, text, favorites, retweets, keyterm, sentiment, date, team) values ('%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s', '%s')" % (df1.datetime[i], df1.username[i], df1.text[i], df1.favorites[i], df1.retweets[i], df1.keyterm[i], df1.sentiment[i], df1.date[i], df1.team[i]))
    cursor.execute(query_insert)

# %% [markdown]
# Abaixo construimos uma função para ler as tabelas do SQL no python.

# %%
def read_table(table):
    query = "SELECT * FROM " + table
    return pd.read_sql(query,con)

# %%
df1 = read_table("ttspl")
# Dado que o processo de armazenamento de uma base faz com suas variáveis passem para o formato 'object', temos que ajustá-las
df1['datetime'] = pd.to_datetime(df1.datetime, format='%Y-%m-%d')
df1['date'] = pd.to_datetime(df1.date, format='%Y-%m-%d')
df1['favorites'] = df1.favorites.astype(int)
df1['retweets'] = df1.retweets.astype(int)
df1

# %%
df1.info()

# %%
# Sempre utilizar este comando para salvar todas as tarefas realizadas no servidor SQL
con.commit()

# %% [markdown]
# Começamos aqui o processo de coleta das informações referentes as partidas disputadas pelos clubes da PL nas últimas temporadas.

# %%
# Criando uma lista de anos
years = list(range(2022,2020,-1))
all_matches = []

# %%
# URL do site
url = "https://fbref.com/en/comps/9/11160/2021-2022-Premier-League-Stats"

# %% [markdown]
# O código tende a dar erros muitas vezes. Isso ocorre pois o site bloqueia o acesso. Por este motivo, recorremos a um tempo de 5 segundos em casa loop. Mesmo assim, bugs podem ocorrer. 

# %%
for year in years:
    data = requests.get(url)
    soup = BeautifulSoup(data.text)
    table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]
    
    previous_season = soup.select("a.prev")[0].get("href")
    url = f"https://fbref.com{previous_season}"
    
    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Premier League"]
        
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches.append(team_data)
        time.sleep(5)

# %%
# Concatenando todas as partidas
match_df = pd.concat(all_matches)

# %%
match_df.columns = [c.lower() for c in match_df.columns]
match_df

# %%
# Criando lista de anos 
years = list(range(2020,2018,-1))
all_matches1 = []

# %%
url1 = "https://fbref.com/en/comps/9/2019-2020/2019-2020-Premier-League-Stats"

# %%
for year in years:
    data = requests.get(url1)
    soup = BeautifulSoup(data.text)
    table = soup.select('table.stats_table')[0]

    links = [l.get("href") for l in table.find_all('a')]
    links = [l for l in links if '/squads/' in l]
    team_urls = [f"https://fbref.com{l}" for l in links]
    
    previous_season = soup.select("a.prev")[0].get("href")
    url1 = f"https://fbref.com{previous_season}"
    
    for team_url in team_urls:
        team_name = team_url.split("/")[-1].replace("-Stats", "").replace("-", " ")
        
        data = requests.get(team_url)
        matches = pd.read_html(data.text, match="Scores & Fixtures")[0]
        soup = BeautifulSoup(data.text)
        links = [l.get("href") for l in soup.find_all('a')]
        links = [l for l in links if l and 'all_comps/shooting/' in l]
        data = requests.get(f"https://fbref.com{links[0]}")
        shooting = pd.read_html(data.text, match="Shooting")[0]
        shooting.columns = shooting.columns.droplevel()
        
        try:
            team_data = matches.merge(shooting[["Date", "Sh", "SoT", "Dist", "FK", "PK", "PKatt"]], on="Date")
        except ValueError:
            continue
        team_data = team_data[team_data["Comp"] == "Premier League"]
        
        team_data["Season"] = year
        team_data["Team"] = team_name
        all_matches1.append(team_data)
        time.sleep(4)

# %%
# Concatenando todas as partidas
match_df1 = pd.concat(all_matches1)

# %%
match_df1.columns = [c.lower() for c in match_df1.columns]
match_df1

# %%
# Concatenando os dois dfs
all_season = pd.concat([match_df,match_df1])

# %%
# Salvando df em formato csv
all_season.to_csv("match_4season.csv")

# %% [markdown]
# Nesta etapa realizamos a raspagem da série do índice SPI, que busca medir a força dos clubes.

# %%
driver = webdriver.Chrome()

# %%
# Entrando no site do FiveThirtyEight
driver.get("https://data.fivethirtyeight.com/#soccer-spi")

# %%
# Clicando no botão de download para baixar as tabelas com os ídnices SPI de cada clube da PL
driver.find_element("xpath", '/html/body/div[2]/div[2]/table[1]/tbody/tr[1]/td[5]/div').click()

# %% [markdown]
# IMPORTANTE: O output do código acima pode variar, dado que a referida tabela do link se ordena de acordo com a atualização das bases. No código, especificamente, o que vai variar será índice do trecho 'tr[1]'.

# %%
# Fechando o driver
driver.quit()

# %%
# Movendo os arquivos zipados para o diretório do código
os.replace("C:\\Users\\guhhh\\Downloads\\soccer-spi.zip", "C:\\Users\\guhhh\\OneDrive\\Área de Trabalho\\Workspace\\modeloPL\\script\\soccer-spi.zip")

# %%
dir = os.getcwd()

# %%
# Deszipando os files que contem a série do SPI
with zipfile.ZipFile('C:\\Users\\guhhh\\OneDrive\\Área de Trabalho\\Workspace\\modeloPL\\script\\soccer-spi.zip', 'r') as zip_ref:
    zip_ref.extractall(dir)

# %%
# https://data.fivethirtyeight.com/#soccer-spi site para baixar o índice SPI
df_spi = pd.read_csv("soccer-spi\\spi_matches.csv")
df_match = pd.read_csv("match_4season.csv")

# %%
# Eliminando temporadas que não serão utilizadas
years = [2016,2017,2022]
df_spi = df_spi[df_spi.season.isin(years)==False]

# %%
#Selecionando apenas os dados da PL
df_spi = df_spi[df_spi["league"]=="Barclays Premier League"]

# %%
# Selecionando as colunas que iremos utilizar no modelo
data =  df_spi[["season","league","date","team1","team2","spi1","spi2","nsxg1","nsxg2"]]
data

# %% [markdown]
# Aqui, filtramos nosso df com as colunas das datas, dos times, do spi e da varíavel nsxg.

# %%
data_team = data[["date","team1","spi1","nsxg1"]]
data_team

# %% [markdown]
# Renomeamos os títulos das colunas retirando o nº 1 do final.

# %%
data_team = data_team.rename(columns={"team1":"team","spi1":"spi","nsxg1":"nsxg"})
data_team

# %% [markdown]
# Aplicamos, aqui, o mesmo procedimento feito acima. Eliminamos o indice 2 dos titulos das colunas renomeandos.

# %%
# Renomeando para fazer o merge.
data_team1 = data[["date","team2","spi2","nsxg2"]]
data_team1 = data_team1.rename(columns={"team2":"team","spi2":"spi","nsxg2":"nsxg"})
data_team1

# %% [markdown]
# Agora, antes de aplicar o merge, concatenamos em uma mesma coluna os elementos referentes aos titulos: date, team, spi e nsxg.

# %%
df_teamC = pd.concat([data_team,data_team1])
df_teamC

# %% [markdown]
# Fazemos então o merge e eliminamos as linhas duplicadas.

# %%
# Fazendo o merge
data_consolidada = pd.merge(df_match,df_teamC,how='inner',on=["date","team"])
data_consolidada = data_consolidada.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
data_consolidada

# %%
matches = data_consolidada.drop_duplicates(subset=['date', 'time', 'opponent', 'team','season'])
matches = matches.drop(['match report', 'round', 'formation', 'notes'], axis=1)
matches.dropna(inplace=True)
matches.reset_index(drop=True, inplace=True)
matches

# %%
# Salvando em formato csv
matches.to_csv("matches.csv")

# %% [markdown]
# Abaixo, vamos armazenar este df em SQL, seguindo o mesmo procedimento realizado com o df dos tts.

# %%
# Utilizando um loop para passar a função de conversão das colunas de texto para um formato compatível com a syntax do SQL
for i in range(0, len(matches)):
    matches.replace(matches.captain[i], text_to_sql(matches.captain[i]), inplace=True)
    matches.replace(matches.referee[i], text_to_sql(matches.referee[i]), inplace=True)
matches.captain

# %%
matches.referee

# %%
cols = matches.columns.tolist()
cols

# %%
query_tb = create_tb('matchespl', cols)
query_tb

# %%
# Criando a tabela no servidor SQL
cursor.execute(query_tb)

# %%
def insert_tb(n, data):
    l = ["INSERT INTO " + str(n) + " ("]
    for i in range(0, (len(data.columns.tolist())-1)):
        q = data.columns.tolist()[i] + ','
        l.append(q)
    l.append(data.columns.tolist()[-1] + ") values (")
    l.append("'%s', "*(len(data.columns.tolist())-1))
    l.append("'%s')")
    query = ' '.join([str(item) for item in l])
    return query

# %%
insert = insert_tb('matchespl', matches)
insert

# %%
# Armazenando o df dos tts
for i in range(0, len(matches)):
    query_insert = insert % (matches.date[i],
                             matches.time[i],
                             matches.comp[i],
                             matches.day[i],
                             matches.venue[i],
                             matches.result[i],
                             matches.gf[i],
                             matches.ga[i],
                             matches.opponent[i],
                             matches.xg[i],
                             matches.xga[i],
                             matches.poss[i],
                             matches.attendance[i],
                             matches.captain[i],
                             matches.referee[i],
                             matches.sh[i],
                             matches.sot[i],
                             matches.dist[i],
                             matches.fk[i],
                             matches.pk[i],
                             matches.pkatt[i],
                             matches.season[i],
                             matches.team[i],
                             matches.spi[i],
                             matches.nsxg[i])
    cursor.execute(query_insert)

# %%
con.commit()

# %%
df1 = read_table("ttspl")
# Dado que o processo de armazenamento em SQL converteu as colunas do df para o formato 'object', temos que ajustá-las
df1['datetime'] = pd.to_datetime(df1.datetime, format='%Y-%m-%d')
df1['date'] = pd.to_datetime(df1.date, format='%Y-%m-%d')
df1['favorites'] = df1.favorites.astype(int)
df1['retweets'] = df1.retweets.astype(int)
df1

# %%
matches = read_table("matchespl")
matches['date'] = pd.to_datetime(matches.date, format='%Y-%m-%d')
matches['gf'] = matches.gf.astype(int)
matches['ga'] = matches.ga.astype(int)
matches['pk'] = matches.pk.astype(float)
matches['pkatt'] = matches.pkatt.astype(float)
matches['poss'] = matches.poss.astype(float)
matches['xg'] = matches.xg.astype(float)
matches['xga'] = matches.xga.astype(float)
matches['attendance'] = matches.attendance.astype(float)
matches['sh'] = matches.sh.astype(float)
matches['sot'] = matches.sot.astype(float)
matches['dist'] = matches.dist.astype(float)
matches['fk'] = matches.fk.astype(float)
matches['spi'] = matches.spi.astype(float)
matches['nsxg'] = matches.nsxg.astype(float)
matches

# %%
matches.info()

# %% [markdown]
# ## Manipulação e Limpeza dos Dados

# %% [markdown]
# Vamos 1º agrupar nosso df dos tts pela data de modo a visualizar o nº de postagens foram feitas por dia, relacionadas aos clubes da pl, em nosso corte temporal.

# %%
df1.groupby('date')['sentiment'].count()

# %% [markdown]
# Os procedimentos abaixo visam agregar nosso df dos tts para que possamos gerar outras variáveis e também, para suavizar as séries temporais contidas nele.

# %%
agg_data1 = (df1.groupby(['date', 'team'])
                 .agg({ 'sentiment':'count',
                        'datetime':'last'}).reset_index())

agg_data = df1.groupby(['date',
                        'team',
                       'sentiment']).agg({'sentiment':'count',
                                        'datetime':'last',}).unstack().reset_index()

agg_data1['positive']= agg_data['sentiment'].positive
agg_data1['negative']= agg_data['sentiment'].negative
agg_data1['neutral'] = agg_data['sentiment'].neutral

# %% [markdown]
# Gerando as taxas de sentimento dos tts, uma variável que representa a diferença entre os tts positivos e negativos de cada clube e as diferenças e médias móveis das séries.

# %%
daily_data = agg_data1
daily_data = daily_data.drop(['datetime'], axis=1)
daily_data = daily_data.rename(columns = {'sentiment':'number_of_tweets'})
daily_data['positive_ratio']=daily_data['positive']/daily_data['number_of_tweets']
daily_data['negative_ratio']=daily_data['negative']/daily_data['number_of_tweets']
daily_data['neutral_ratio']=daily_data['neutral']/daily_data['number_of_tweets']
daily_data['gap_sent'] = daily_data['positive_ratio'] - daily_data['negative_ratio']
daily_data['diff_positive_ratio'] = daily_data.positive_ratio.diff()
daily_data['var_diff_positive_ratio'] = daily_data.diff_positive_ratio.diff()
daily_data['diff_negative_ratio'] = daily_data.negative_ratio.diff()
daily_data['var_diff_negative_ratio'] = daily_data.diff_negative_ratio.diff()
daily_data['diff_neutral_ratio'] = daily_data.neutral_ratio.diff()
daily_data['var_diff_neutral_ratio'] = daily_data.diff_neutral_ratio.diff()
daily_data['diff_gap_sent'] = daily_data.gap_sent.diff()
daily_data['var_diff_gap_sent'] = daily_data.diff_gap_sent.diff()
daily_data['ma_positive_ratio'] = daily_data['diff_positive_ratio'].rolling(7).sum()
daily_data['ma_negative_ratio'] = daily_data['diff_negative_ratio'].rolling(7).sum()
daily_data['ma_neutral_ratio'] = daily_data['diff_neutral_ratio'].rolling(7).sum()
daily_data['ma_gap_sent'] = daily_data['diff_gap_sent'].rolling(7).sum()
daily_data['date'] = pd.to_datetime(daily_data.date, format='%Y-%m-%d')
daily_data

# %%
daily_data.info()

# %% [markdown]
# Aqui, tiramos a média da váriavel de diferença entre os tts positivos e negativos de cada clube, e as ordenamos de modo a visualizar quais deles apresentam os maiores e quais apresentam os menores valores.

# %%
daily_data.groupby('team')['gap_sent'].mean().sort_values()

# %% [markdown]
# Aqui nós criamos uma função para gerar uma coluna adicional em nosso df das partidas de modo a gerar saidas que indiquem o vencedor em cada uma delas.

# %%
def label(data):
    if data["gf"] > data["ga"]:
        return data["team"]
    elif data["ga"] > data["gf"]:
        return data["opponent"]
    elif data["gf"] == data["ga"]:
        return "Draw"
matches["winner"] = matches.apply(lambda matches:label(matches),axis=1)
matches

# %% [markdown]
# Os dois códigos abaixo tem por objetivo criar um df auxiliar com o nº de gols por time, para que possa ser criada uma visualização sobre isto mais a frente.

# %%
teams = matches.team.unique().tolist()
teams

# %%
l = []
for i in range(0, len(teams)):
    f = {}
    team = teams[i]
    gol = matches[matches.team == teams[i]].gf.sum() + matches[matches.opponent == teams[i]].ga.sum()
    f['team'] = team
    f['goals'] = gol
    l.append(f)
goals = pd.DataFrame(l)
goals = goals.sort_values('goals', ascending=False)
goals = goals.reset_index()
goals = goals.drop(['index'], axis=1)
goals

# %% [markdown]
# Nestas quatro linhas de código subsequentes, objetivamos criar um objeto 'networkx.classes' e outro df auxiliar com o nº de partidas que cada time jogou com cada um dos outros times da amostra, para também gerar outras duas visualizações mais a frente.

# %%
v = matches[['team', 'opponent']]
v

# %%
g = nx.from_pandas_edgelist(v,"team","opponent")
g

# %%
i = matches["winner"].value_counts()[1:50].index
c = matches[(matches["team"].isin(i)) & (matches["opponent"].isin(i))]
d = pd.crosstab(c["team"],c["opponent"])
d

# %% [markdown]
# Já as operações abaixo tem como objetivo criar um df auxiliar normalizado, com intervalo de 0 a 100, de modo a criar a função de comparabilidade dos times através de uma visualização.

# %%
num_cols = ['xg', 'xga', 'poss', 'attendance', 'sh', 'sot', 'dist', 'fk', 'pk', 'pkatt', 'spi', 'nsxg']
n = matches.groupby("team")[num_cols].mean().reset_index()
c = n.drop(['team'], axis=1)
c

# %%
scaler = MinMaxScaler(feature_range=(0, 100))
names = c.columns
sc = scaler.fit_transform(c)
scaled_df = pd.DataFrame(sc, columns=names)
scaled_df.head()

# %%
n = n.drop(c.columns.tolist(), axis=1)
n = pd.concat([n, scaled_df], axis=1, join='inner')
n['xga'] = (n.xga - 100)*-1
n['dist'] = (n.dist - 100)*-1
n

# %% [markdown]
# ## Visualização/Análise dos Dados
# 
# 

# %% [markdown]
# Nessa sessão, elaboramos uma série de visualizações que permitem análises gráficas dos nossos dados.

# %% [markdown]
# Começamos criando dois gráficos de círculo, cada qual mostrando a distribuição dos sentimentos em nosso df dos tts. O 1º é apenas para verificar como tal distribuição varia caso filtremos nosso df para os tts com nº de curtidas maior ou igual a 50. No segundo, consideramos nossa base completa. 

# %%
df1[df1.favorites >= 50]['sentiment'].value_counts().plot.pie(autopct = "%1.0f%%", # Mostrar o valor percentual
                                             colors =sns.color_palette("icefire",3), #icefire
                                             figsize=(9,9),
                                             wedgeprops = {"linewidth":2,"edgecolor":"white"})

my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.title("Sentiment distribution in tweet")
plt.show()

# %%
df1['sentiment'].value_counts().plot.pie(autopct = "%1.0f%%", # Mostrar o valor percentual
                                             colors =sns.color_palette("icefire",3), #icefire
                                             figsize=(9,9),
                                             wedgeprops = {"linewidth":2,"edgecolor":"white"})

my_circ = plt.Circle((0,0),.7,color = "white")
plt.gca().add_artist(my_circ)
plt.title("Sentiment distribution in tweet")
plt.show()

# %% [markdown]
# Ainda no mérito da variável sentimento, num segundo momento, apresentamos um gráfico de linha mostrando a variação temporal da variável 'gap_sent' para o Manchester City.

# %%
daily_data.loc[(daily_data['team']=='Manchester City')].set_index('date')[['ma_positive_ratio']].plot(figsize=(15,9))

# %% [markdown]
# Em seguida, através de uma função de visualização, criamos um word cloud com o nome as palavras mais presentes no tweets de nossa base.

# %%
def PL_WordCloud(data,sentiment,image):
    words= []
    stopwords = nltk.corpus.stopwords.words("english")
    for i in data['text'][ data['sentiment']==sentiment].to_list():
        token = nltk.word_tokenize(i)
        for word in token:
            words.append(word)
    words = [ w for w in words if w.isalpha() == True]
    words= [ w for w in words if w not in stopwords]
    wrd = pd.DataFrame(words)
    wrd = wrd[0].unique()
    img = np.array(Image.open("C:\\Users\\guhhh\\OneDrive\\Área de Trabalho\\Workspace\\modeloPL\\image\\"+image+".jpg"))
    wc = WordCloud(background_color="white",scale=2,mask=img,colormap="Dark2",max_words=100).generate(" ".join(wrd))
    fig = plt.figure(figsize=(12,9))
    plt.imshow(wc,interpolation="bilinear")
    plt.axis("off")
    plt.title("Word Cloud Related to the {} ".format(sentiment))
    plt.show()

# %%
PL_WordCloud(df1, 'positive', 'PL_logo')

# %% [markdown]
# Abaixo o gráfico de barra com o nº de gols por clube.

# %%
plt.figure(figsize=(9,14))
ax = sns.barplot(x="goals",y="team",
                 data=goals,palette="rainbow",
                linewidth = 1,edgecolor = ['k' for i in range(0,24)])

for i,j in enumerate(goals["goals"][:24]):
    ax.text(20,i,j,weight="bold",color = "k",fontsize =12)

plt.title("Gols por Time da PL")
plt.show()

# %% [markdown]
# Aqui seguem as duas visualizações referentes as partidas jogadas entre os clubes de nossa base.
# A 1ª consiste num gráfico de teia, onde cada linha representa uma partida. Já a 2ª corresponde a um gráfico de calor, com cada cédula representando a quantidade de partidas disputadas entre os respectivos times dos eixos.

# %%
fig = plt.figure(figsize=(10,10))
nx.draw_kamada_kawai(g,with_labels =True,node_size =1500,node_color ="Purple",alpha=.8)
plt.title("Teia de Jogos Entre os Times da PL")
fig.set_facecolor("white")
plt.show()

# %%
plt.figure(figsize=(13,10))
sns.heatmap(d,annot=True,cmap=sns.color_palette("inferno")) # Opção annot
plt.title("Número de Jogos entre os Times da PL")
plt.show()

# %% [markdown]
# Por fim, segue nossa função de comparação dos times, construida apartir do df auxiliar normalizado criado acima.

# %%
def team_comparator(team1,team2):
    
    team_list = [team1,team2]
    length    = len(team_list)
    cr        = ["b","r"]
    fig = plt.figure(figsize=(9,9))
    plt.subplot(111,projection= "polar")
    
    for i,j,k in zip(team_list,range(length),cr):
        cats = num_cols
        N    = len(cats)
        
        values = n[n["team"] ==  i][cats].values.flatten().tolist()
        values += values[:1]
        
        angles = [n/float(N)*2*pi for n in range(N)]
        angles += angles[:1]
        
        plt.xticks(angles[:-1],cats,color="k",fontsize=15)
        plt.plot(angles,values,linewidth=3,color=k)
        plt.fill(angles,values,color = k,alpha=.4,label = i)
        plt.legend(loc="best",frameon =True,prop={"size":15}).get_frame().set_facecolor("lightgrey")
        fig.set_facecolor("w")
        fig.set_edgecolor("k")
        plt.title("Comparador dos Times da PL",fontsize=30,color="tomato")

# %%
team_comparator('Manchester City','Arsenal')

# %% [markdown]
# ## Construção do Modelo Preditivo

# %% [markdown]
# Num primeiro momento, ordenamos nosso df das partidas pela data, de modo a visualizar o ponto inicial e o ponto final de nosso corte de tempo.

# %%
matches.sort_values('date')

# %% [markdown]
# Em sequência, criamos nossa variável de interesse e ajustamos alguns de nossos preditores, criando variáveis adicionais categóricas e compatibilizadas.

# %%
matches["target"] = (matches["result"] == "W").astype("int")
matches["venue_code"] = matches["venue"].astype("category").cat.codes
matches["opp_code"] = matches["opponent"].astype("category").cat.codes
matches["hour"] = matches["time"].str.replace(":.+", "", regex=True).astype("int")
matches["day_code"] = matches["date"].dt.dayofweek
matches

# %% [markdown]
# Aqui, criamos um objeto com nosso modelo de machine learning. Usaremos o método "Random Forest" para treinar o modelo.

# %%
rf = RandomForestClassifier(n_estimators=50, min_samples_split=10, random_state=1)

# %% [markdown]
# Criamos aqui duas sub amostras, sendo uma para treino e outra para teste do nosso modelo, dividindo nosso df entre observações anteriores e observações posteriores ao 1º dia do ano de 2022.

# %%
train = matches[matches["date"] < '2022-01-01']
test = matches[matches["date"] > '2022-01-01']
predictors = ["venue_code", "opp_code", "hour", "day_code", "spi", "nsxg"]

# %% [markdown]
# Rodamos então nosso modelo.

# %%
rf.fit(train[predictors], train["target"])

# %%
preds = rf.predict(test[predictors])

# %%
error = accuracy_score(test["target"], preds)
error

# %% [markdown]
# Em seu resultado preeliminar, ele apresenta uma acurácia de 67,9%.

# %%
combined = pd.DataFrame(dict(actual=test["target"], predicted=preds))
pd.crosstab(index=combined["actual"], columns=combined["predicted"])

# %%
precision_score(test["target"], preds)

# %% [markdown]
# Testando alguns ajustes.

# %%
grouped_matches = matches.groupby("team")
group = grouped_matches.get_group("Manchester City").sort_values("date")
group

# %%
def rolling_averages(group, cols, new_cols):
    group = group.sort_values("date")
    rolling_stats = group[cols].rolling(3, closed='left').mean()
    group[new_cols] = rolling_stats
    group = group.dropna(subset=new_cols)
    return group

# %%
cols = ["gf", "ga", "sh", "sot", "dist", "fk", "pk", "pkatt", "spi", "nsxg"]
new_cols = [f"{c}_rolling" for c in cols]

rolling_averages(group, cols, new_cols)

# %%
matches_rolling = matches.groupby("team").apply(lambda x: rolling_averages(x, cols, new_cols))
matches_rolling = matches_rolling.droplevel('team')
matches_rolling.index = range(matches_rolling.shape[0])
matches_rolling

# %%
def make_predictions(data, predictors):
    train = data[data["date"] < '2022-01-01']
    test = data[data["date"] > '2022-01-01']
    rf.fit(train[predictors], train["target"])
    preds = rf.predict(test[predictors])
    combined = pd.DataFrame(dict(actual=test["target"], predicted=preds), index=test.index)
    error = precision_score(test["target"], preds)
    return combined, error
combined, error = make_predictions(matches_rolling, predictors + new_cols)
error

# %%
combined = combined.merge(matches_rolling[["date", "team", "opponent", "result"]], left_index=True, right_index=True)
combined.head(10)

# %%
class MissingDict2(dict):
    __missing__ = lambda self, key: key

map_values2 = {"Brighton and Hove Albion": "Brighton",
              "Manchester United": "Manchester Utd",
              "Newcastle United": "Newcastle Utd",
              "Tottenham Hotspur": "Tottenham",
              "West Ham United": "West Ham",
              "Wolverhampton Wanderers": "Wolves"} 
mapping2 = MissingDict2(**map_values2)

# %%
combined["new_team"] = combined["team"].map(mapping2)
merged = combined.merge(combined, left_on=["date", "new_team"], right_on=["date", "opponent"])
merged

# %%
merged[(merged["predicted_x"] == 1) & (merged["predicted_y"] == 0)]["actual_x"].value_counts()

# %%
47/74

# %% [markdown]
# Por fim, a maior acurácia em encontrada no modelo sem ajustes.


