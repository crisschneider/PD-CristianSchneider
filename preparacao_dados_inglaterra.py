# carregando pandas para parsear csv's
import pandas as pd
# carregando numpy para ações matemáticas
import numpy as np
# funções matemáticas
import math

# leitura dos CSVs
# temporadas para inicialização do ELO
d_1 = pd.read_csv("../SoccerPrediction/Data_England/00.01.csv")
d_2 = pd.read_csv("../SoccerPrediction/Data_England/01.02.csv")
# temporadas de treinamento
d_3 = pd.read_csv("../SoccerPrediction/Data_England/02.03.csv")
d_4 = pd.read_csv("../SoccerPrediction/Data_England/03.04.csv")
d_5 = pd.read_csv("../SoccerPrediction/Data_England/04.05.csv")
d_6 = pd.read_csv("../SoccerPrediction/Data_England/05.06.csv")
d_7 = pd.read_csv("../SoccerPrediction/Data_England/06.07.csv")
d_8 = pd.read_csv("../SoccerPrediction/Data_England/07.08.csv")
d_9=pd.read_csv("../SoccerPrediction/Data_England/08.09.csv")
d_10=pd.read_csv("../SoccerPrediction/Data_England/09.10.csv")
d_11=pd.read_csv("../SoccerPrediction/Data_England/10.11.csv")
d_12=pd.read_csv("../SoccerPrediction/Data_England/11.12.csv")
d_13=pd.read_csv("../SoccerPrediction/Data_England/12.13.csv")
d_14=pd.read_csv("../SoccerPrediction/Data_England/13.14.csv")
d_15=pd.read_csv("../SoccerPrediction/Data_England/14.15.csv")
d_16=pd.read_csv("../SoccerPrediction/Data_England/15.16.csv")
d_17=pd.read_csv("../SoccerPrediction/Data_England/16.17.csv")
d_18=pd.read_csv("../SoccerPrediction/Data_England/17.18.csv")

# criando dataframes
# temporadas para inicialização do ELO
df_1 = pd.DataFrame(d_1, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_2 = pd.DataFrame(d_2, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_3 = pd.DataFrame(d_3, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
# temporadas de treinamento
df_4 = pd.DataFrame(d_4, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_5 = pd.DataFrame(d_5, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_6 = pd.DataFrame(d_6, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_7 = pd.DataFrame(d_7, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_8 = pd.DataFrame(d_8, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_9 = pd.DataFrame(d_9, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_10 = pd.DataFrame(d_10, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_11 = pd.DataFrame(d_11, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_12 = pd.DataFrame(d_12, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_13 = pd.DataFrame(d_13, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_14 = pd.DataFrame(d_14, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_15 = pd.DataFrame(d_15, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_16 = pd.DataFrame(d_16, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_17 = pd.DataFrame(d_17, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])
df_18 = pd.DataFrame(d_18, columns = ['HomeTeam','AwayTeam','FTHG','FTAG','FTR'])


#convertendo o resultado do tipo 0-0 para D
df_1['FTR'] = np.where((df_1['FTHG'] < df_1['FTAG']), 2, 1)
df_1['FTR'] = np.where((df_1['FTHG'] == df_1['FTAG']), 0, df_1['FTR'])

df_2['FTR'] = np.where((df_2['FTHG'] < df_2['FTAG']), 2, 1)
df_2['FTR'] = np.where((df_2['FTHG'] == df_2['FTAG']), 0, df_2['FTR'])

df_3['FTR'] = np.where((df_3['FTHG'] < df_3['FTAG']), 2, 1)
df_3['FTR'] = np.where((df_3['FTHG'] == df_3['FTAG']), 0, df_3['FTR'])

df_4['FTR'] = np.where((df_4['FTHG'] < df_4['FTAG']), 2, 1)
df_4['FTR'] = np.where((df_4['FTHG'] == df_4['FTAG']), 0, df_4['FTR'])

df_5['FTR'] = np.where((df_5['FTHG'] < df_5['FTAG']), 2, 1)
df_5['FTR'] = np.where((df_5['FTHG'] == df_5['FTAG']), 0, df_5['FTR'])

df_6['FTR'] = np.where((df_6['FTHG'] < df_6['FTAG']), 2, 1)
df_6['FTR'] = np.where((df_6['FTHG'] == df_6['FTAG']), 0, df_6['FTR'])

df_7['FTR'] = np.where((df_7['FTHG'] < df_7['FTAG']), 2, 1)
df_7['FTR'] = np.where((df_7['FTHG'] == df_7['FTAG']), 0, df_7['FTR'])

df_8['FTR'] = np.where((df_8['FTHG'] < df_8['FTAG']), 2, 1)
df_8['FTR'] = np.where((df_8['FTHG'] == df_8['FTAG']), 0, df_8['FTR'])

df_9['FTR'] = np.where((df_9['FTHG'] < df_9['FTAG']), 2, 1)
df_9['FTR'] = np.where((df_9['FTHG'] == df_9['FTAG']), 0, df_9['FTR'])

df_10['FTR'] = np.where((df_10['FTHG'] < df_10['FTAG']), 2, 1)
df_10['FTR'] = np.where((df_10['FTHG'] == df_10['FTAG']), 0, df_10['FTR'])

df_11['FTR'] = np.where((df_11['FTHG'] < df_11['FTAG']), 2, 1)
df_11['FTR'] = np.where((df_11['FTHG'] == df_11['FTAG']), 0, df_11['FTR'])

df_12['FTR'] = np.where((df_12['FTHG'] < df_12['FTAG']), 2, 1)
df_12['FTR'] = np.where((df_12['FTHG'] == df_12['FTAG']), 0, df_12['FTR'])

df_13['FTR'] = np.where((df_13['FTHG'] < df_13['FTAG']), 2, 1)
df_13['FTR'] = np.where((df_13['FTHG'] == df_13['FTAG']), 0, df_13['FTR'])

df_14['FTR'] = np.where((df_14['FTHG'] < df_14['FTAG']), 2, 1)
df_14['FTR'] = np.where((df_14['FTHG'] == df_14['FTAG']), 0, df_14['FTR'])

df_15['FTR'] = np.where((df_15['FTHG'] < df_15['FTAG']), 2, 1)
df_15['FTR'] = np.where((df_15['FTHG'] == df_15['FTAG']), 0, df_15['FTR'])

df_16['FTR'] = np.where((df_16['FTHG'] < df_16['FTAG']), 2, 1)
df_16['FTR'] = np.where((df_16['FTHG'] == df_16['FTAG']), 0, df_16['FTR'])

df_17['FTR'] = np.where((df_17['FTHG'] < df_17['FTAG']), 2, 1)
df_17['FTR'] = np.where((df_17['FTHG'] == df_17['FTAG']), 0, df_17['FTR'])

df_18['FTR'] = np.where((df_18['FTHG'] < df_18['FTAG']), 2, 1)
df_18['FTR'] = np.where((df_18['FTHG'] == df_18['FTAG']), 0, df_18['FTR'])


# criando um dataframe com todos os times que jogam as temporadas 05/06 e 06/07
df_unique = pd.DataFrame(df_1['HomeTeam'].unique())
# inicializando todos os times com 1500 de ELO
df_unique['ELO'] = 1500
# nomeando as colunas
df_unique.columns = ['Team', 'ELO']


# função que calcula o ELO
def get_new_scores(home_elo,away_elo,fthg,ftag,ftr):
    # fthg = full time hometeam goals
    # ftag = full time away goals
    # ftr = full time result
    # constante para premier league
    K = 30
    G = 0
    W_h = 0
    W_a = 0

    # diferença de elo entre os times do ponto de vista do time mandante
    rating_diff_h = home_elo - away_elo

    # o resultado esperado é baseado no ponto de vista do time com maior elo
    home_expected = 1 / (1 + 10**(-(rating_diff_h+100)/400))
    away_expected = 1 - home_expected

    # calculando o parâmetro G e W
    if ftr.item() == 1:
        goals_diff = fthg.item() - ftag.item()
        W_h = 1
        W_a = 0

        if goals_diff == 1:
            G = 1

        else:
            G = math.log10(1.7*goals_diff)*2/(2+0.001*rating_diff_h)

    elif ftr.item() == 2:
        goals_diff = ftag.item() - fthg.item()
        W_h = 0
        W_a = 1

        if goals_diff == 1:
            G = 1

        else:
            G = math.log10(1.7*goals_diff)*2/(2+0.001*rating_diff_h)

    elif ftr.item() == 0:
        W_h = 0.5
        W_a = 0.5
        G = 1

    # calculando os novos elos
    home_new_elo = home_elo + K*G*(W_h - home_expected)
    away_new_elo = away_elo + K*G*(W_a - away_expected)

    return home_new_elo,away_new_elo

# função que itera por todas as temporadas e calcula o ELO de cada time em cada momento
def calculate_elo(df):

    for index, row in df.iterrows():

        # inicializando as variáveis locais
        old_home_elo = None
        old_away_elo = None
        home_team = None
        away_team = None

        # variáveis de interesse para o cálculo do ELO
        # FTHG = full time hometeam goals
        # FTAG = full time away goals
        # FTR = full time result
        h = df.loc[index,'FTHG']
        a = df.loc[index,'FTAG']
        f = df.loc[index,'FTR']

        # identificando o mandante e o visitante da linha em questão
        home_team = df.loc[index,'HomeTeam']
        away_team = df.loc[index,'AwayTeam']

        # procurando o elo antigo dos times no dataframe com times e elos
        # salvando os elos antigos em variáveis
        old_home_elo = df_unique.loc[df_unique.Team == home_team, 'ELO'].item()
        old_away_elo = df_unique.loc[df_unique.Team == away_team, 'ELO'].item()
        # salvando no dataframe em uma nova coluna
        df.loc[index, 'OAE'] = old_away_elo
        df.loc[index, 'OHE'] = old_home_elo
        df.loc[index, 'ED'] = old_home_elo - old_away_elo


        # calculando o novo elo
        new_home_elo,new_away_elo = get_new_scores(old_home_elo,old_away_elo,h,a,f)
        # salvando no dataframe em uma nova coluna
        #df.loc[index, 'new away ELO'] = new_away_elo
        #df.loc[index, 'new home ELO'] = new_home_elo

        # guardando o novo elo dos times no dataframe com times e elos
        df_unique.loc[df_unique.Team == home_team, 'ELO'] = new_home_elo
        df_unique.loc[df_unique.Team == away_team, 'ELO'] = new_away_elo

# função que atualiza o dataframe que guarda os times que estão disputando
# a temporada analisada
def update_teams(eigheenth, nineteenth, twentieth, first, second, third):

    # trazendo os ELOs de volta para a media
    for index, row in df_unique.iterrows():

        df_unique.loc[index,'ELO'] = df_unique.loc[index,'ELO']*0.8 + 0.2*1500

    # identificando os times rebaixados
    elo_eigheenth = df_unique.loc[df_unique.Team == eigheenth, 'ELO'].item()
    elo_nineteenth = df_unique.loc[df_unique.Team == nineteenth, 'ELO'].item()
    elo_twentieth = df_unique.loc[df_unique.Team == twentieth, 'ELO'].item()

    # somando seus ELOs
    sum_elo_relegated = elo_eigheenth + elo_nineteenth + elo_twentieth
    mean_elo_relegated = sum_elo_relegated/3

    # corrigindo os ELOs
    for index, row in df_unique.iterrows():

        team = df_unique.loc[index,'Team']

        # times que continuam na Premier League
        if team == eigheenth or team == nineteenth or team == twentieth:

            df_unique.loc[index,'ELO'] = mean_elo_relegated

    # times rebaixados
    # eigheenth = decimo oitavo colocado da Premier league
    # nineteenth = decimo nono colocado da Premier league
    # twentieth = vigésimo colocado da Premier league

    # times promovidos
    # first = primeiro colocado da Football league
    # second = segundo colocado da Football league
    # third = terceiro colcoado da Football league

    df_unique.loc[df_unique.Team == eigheenth, 'Team'] = first
    df_unique.loc[df_unique.Team == nineteenth, 'Team'] = second
    df_unique.loc[df_unique.Team == twentieth, 'Team'] = third


def calculate_form(df):

    # definindo um dataframe local contendo todos os times desta temporada
    df_unique_local =  pd.DataFrame(df['HomeTeam'].unique())
    df_unique_local.columns = ['Team']

    # inicializando variáveis locais para forma
    df['AF'] = 0      # awayteam form
    df['HF'] = 0      # hometeam form
    df['Used'] = 0    # Used = 1 -> utilizado para previsões

    # iterando por todos os times da temporada
    for i, r in df_unique_local.iterrows():

        # inicializando variáveis locais
        team = df_unique_local.loc[i,'Team']
        match_result = 0
        coeff_n_minus_7 = 0
        coeff_n_minus_6 = 0
        coeff_n_minus_5 = 0
        coeff_n_minus_4 = 0
        coeff_n_minus_3 = 0
        coeff_n_minus_2 = 0
        coeff_n_minus_1 = 0

        # todas as partidas que envolvem o time dessa iteração
        df_temp = df.loc[(df.HomeTeam == team) | (df.AwayTeam == team)]
        df_temp = df_temp.reset_index()

        # loop onde será calculada a forma e armazenada em um dataframe temp
        for index, row in df_temp.iterrows():

            hometeam = df_temp.loc[index, 'HomeTeam']
            awayteam = df_temp.loc[index, 'AwayTeam']

            # as N primeiras partidas são ignoradas
            if index > 7:

                # levantando os dados das últimas quatro partidas
                result_n_minus_7 = df_temp.loc[(index - 7),'FTR'].item()
                result_n_minus_6 = df_temp.loc[(index - 6),'FTR'].item()
                result_n_minus_5 = df_temp.loc[(index - 5),'FTR'].item()
                result_n_minus_4 = df_temp.loc[(index - 4),'FTR'].item()
                result_n_minus_3 = df_temp.loc[(index - 3),'FTR'].item()
                result_n_minus_2 = df_temp.loc[(index - 2),'FTR'].item()
                result_n_minus_1 = df_temp.loc[(index - 1),'FTR'].item()
                n_minus_7_hometeam = df_temp.loc[(index - 7),'HomeTeam']
                n_minus_6_hometeam = df_temp.loc[(index - 6),'HomeTeam']
                n_minus_5_hometeam = df_temp.loc[(index - 5),'HomeTeam']
                n_minus_4_hometeam = df_temp.loc[(index - 4),'HomeTeam']
                n_minus_3_hometeam = df_temp.loc[(index - 3),'HomeTeam']
                n_minus_2_hometeam = df_temp.loc[(index - 2),'HomeTeam']
                n_minus_1_hometeam = df_temp.loc[(index - 1),'HomeTeam']
                n_minus_7_awayteam = df_temp.loc[(index - 7),'AwayTeam']
                n_minus_6_awayteam = df_temp.loc[(index - 6),'AwayTeam']
                n_minus_5_awayteam = df_temp.loc[(index - 5),'AwayTeam']
                n_minus_4_awayteam = df_temp.loc[(index - 4),'AwayTeam']
                n_minus_3_awayteam = df_temp.loc[(index - 3),'AwayTeam']
                n_minus_2_awayteam = df_temp.loc[(index - 2),'AwayTeam']
                n_minus_1_awayteam = df_temp.loc[(index - 1),'AwayTeam']

                # inicializando variáveis locais
                coeff_n_minus_7 = 0
                coeff_n_minus_6 = 0
                coeff_n_minus_5 = 0
                coeff_n_minus_4 = 0
                coeff_n_minus_3 = 0
                coeff_n_minus_2 = 0
                coeff_n_minus_1 = 0

                # partida n-7
                if result_n_minus_7 == 0:

                    coeff_n_minus_7 = 1

                elif (result_n_minus_7 == 1 and n_minus_7_hometeam == team) or (result_n_minus_7 == 2 and n_minus_7_awayteam == team):

                    coeff_n_minus_7 = 7

                else:

                    coeff_n_minus_7 = 0

                # partida n-6
                if result_n_minus_6 == 0:

                    coeff_n_minus_6 = 1

                elif (result_n_minus_6 == 1 and n_minus_6_hometeam == team) or (result_n_minus_6 == 2 and n_minus_6_awayteam == team):

                    coeff_n_minus_6 = 6

                else:

                    coeff_n_minus_6 = 0

                # partida n-5
                if result_n_minus_5 == 0:

                    coeff_n_minus_5 = 1

                elif (result_n_minus_5 == 1 and n_minus_5_hometeam == team) or (result_n_minus_5 == 2 and n_minus_5_awayteam == team):

                    coeff_n_minus_5 = 5

                else:

                    coeff_n_minus_5 = 0

                # partida n-4
                if result_n_minus_4 == 0:

                    coeff_n_minus_4 = 1

                elif (result_n_minus_4 == 1 and n_minus_4_hometeam == team) or (result_n_minus_4 == 2 and n_minus_4_awayteam == team):

                    coeff_n_minus_4 = 4

                else:

                    coeff_n_minus_4 = 0

                # partida n-3
                if result_n_minus_3 == 0:

                    coeff_n_minus_3 = 1

                elif (result_n_minus_3 == 1 and n_minus_3_hometeam == team) or (result_n_minus_3 == 2 and n_minus_3_awayteam == team):

                    coeff_n_minus_3 = 3

                else:

                    coeff_n_minus_3 = 0

                # partida n-2
                if result_n_minus_2 == 0:

                    coeff_n_minus_2 = 1

                elif (result_n_minus_2 == 1 and n_minus_2_hometeam == team) or (result_n_minus_2 == 2 and n_minus_2_awayteam == team):

                    coeff_n_minus_2 = 3


                else:

                    coeff_n_minus_2 = 0

                # partida n-1
                if result_n_minus_1 == 0:

                    coeff_n_minus_1 = 1

                elif (result_n_minus_1 == 1 and n_minus_1_hometeam == team) or (result_n_minus_1 == 2 and n_minus_1_awayteam == team):

                    coeff_n_minus_1 = 3

                else:

                    coeff_n_minus_1 = 0


                form = 1*coeff_n_minus_7 + 2*coeff_n_minus_6 + 3*coeff_n_minus_5 + 4*coeff_n_minus_4 + 5*coeff_n_minus_3 + 6*coeff_n_minus_2 + 7*coeff_n_minus_1

                temp_index = df.index[(df.HomeTeam == hometeam) & (df.AwayTeam == awayteam)]

                hometeam_temp = df.loc[temp_index, 'HomeTeam'].item()
                awayteam_temp = df.loc[temp_index, 'AwayTeam'].item()

                if hometeam_temp == team:

                    df.loc[temp_index, 'HF'] = form

                elif awayteam_temp == team:

                    df.loc[temp_index, 'AF'] = form

                if hometeam_temp == team and index > 18:

                    df.loc[temp_index, 'Used'] = 1

                elif awayteam_temp == team and index > 18:

                    df.loc[temp_index, 'Used'] = 1

def calculate_past_games_features(df):

    # definindo um dataframe local contendo todos os times desta temporada
    df_unique_local =  pd.DataFrame(df['HomeTeam'].unique())
    df_unique_local.columns = ['Team']

    # inicializando features
    df['HHGR'] = 0  # hometeam_home_goals_ratio
    df['HSHGR'] = 0 # hometeam_suffered_home_goals_ratio
    df['AAGR'] = 0  # awayteam_away_goals_ratio
    df['ASAGR'] = 0 # awayteam_suffered_away_goals_ratio
    df['VHHR'] = 0  # victories hometeam home ratio
    df['DHHR'] = 0  # draws hometeam home ratio
    df['LHHR'] = 0  # losses hometeam home ratio
    df['VAAR'] = 0  # victories awayteam away ratio
    df['DAAR'] = 0  # draws awayteam away ratiio
    df['LAAR'] = 0  # losses awayteam away ratio

    # iterando por todos os times da temporada
    for i, r in df_unique_local.iterrows():

        # inicializando variáveis
        team = df_unique_local.loc[i,'Team']
        counter_home_matches = 1
        counter_away_matches = 1
        home_goal = 0
        away_goal = 0
        home_goal_ratio = 0
        away_goal_ratio = 0
        last_home_goal = 0
        last_away_goal = 0
        suff_home_goal = 0
        suff_away_goal = 0
        suff_home_goal_ratio = 0
        suff_away_goal_ratio = 0
        last_suff_home_goal = 0
        last_suff_away_goal = 0
        victories_hometeam_home = 0
        losses_hometeam_home = 0
        draws_hometeam_home = 0
        victories_hometeam_home_ratio = 0
        losses_hometeam_home_ratio = 0
        draws_hometeam_home_ratio = 0
        last_victories_hometeam_home = 0
        last_losses_hometeam_home = 0
        last_draws_hometeam_home = 0
        victories_awayteam_away = 0
        losses_awayteam_away = 0
        draws_awayteam_away = 0
        victories_awayteam_away_ratio = 0
        losses_awayteam_away_ratio = 0
        draws_awayteam_away_ratio = 0
        last_victories_awayteam_away = 0
        last_losses_awayteam_away = 0
        last_draws_awayteam_away = 0
        match_result = 0

        # todas as partidas que envolvem o time dessa iteração
        df_temp = df.loc[(df.HomeTeam == team) | (df.AwayTeam == team)]
        df_temp = df_temp.reset_index()

        # loop onde será calculada a forma e armazenada em um dataframe temp
        for index, row in df_temp.iterrows():

            hometeam = df_temp.loc[index, 'HomeTeam']
            awayteam = df_temp.loc[index, 'AwayTeam']

            # cálculo das médias de gols
            if hometeam == team:

                # primeira iteração da temporada
                if index == 0:

                    home_goal = df_temp.loc[index, 'FTHG'].item()
                    suff_home_goal = df_temp.loc[index, 'FTAG'].item()
                    home_goal_ratio = home_goal/counter_home_matches
                    suff_home_goal_ratio = suff_home_goal/counter_home_matches

                    match_result = df_temp.loc[index, 'FTR'].item()

                    if match_result == 1:

                        victories_hometeam_home = 1
                        victories_hometeam_home_ratio = 1

                    elif match_result == 2:

                        losses_hometeam_home = 1
                        losses_hometeam_home_ratio = 1

                    elif match_result == 0:

                        draws_hometeam_home = 1
                        draws_hometeam_home_ratio = 1

                elif index > 0:

                    home_goal = last_home_goal + df_temp.loc[index, 'FTHG'].item()
                    suff_home_goal = last_suff_home_goal + df_temp.loc[index, 'FTAG'].item()
                    home_goal_ratio = home_goal/counter_home_matches
                    suff_home_goal_ratio = suff_home_goal/counter_home_matches

                    match_result = df_temp.loc[index, 'FTR'].item()

                    if match_result == 1:

                        victories_hometeam_home = last_victories_hometeam_home + 1
                        victories_hometeam_home_ratio = victories_hometeam_home/counter_home_matches
                        losses_hometeam_home_ratio = losses_hometeam_home/counter_home_matches
                        draws_hometeam_home_ratio = draws_hometeam_home/counter_home_matches

                    elif match_result == 2:

                        losses_hometeam_home = last_losses_hometeam_home + 1
                        losses_hometeam_home_ratio = losses_hometeam_home/counter_home_matches
                        victories_hometeam_home_ratio = victories_hometeam_home/counter_home_matches
                        draws_hometeam_home_ratio = draws_hometeam_home/counter_home_matches

                    elif match_result == 0:

                        draws_hometeam_home = last_draws_hometeam_home + 1
                        draws_hometeam_home_ratio = draws_hometeam_home/counter_home_matches
                        victories_hometeam_home_ratio = victories_hometeam_home/counter_home_matches
                        losses_hometeam_home_ratio = losses_hometeam_home/counter_home_matches

                # atualizando variáveis para próxima iteração
                counter_home_matches += 1
                last_home_goal = home_goal
                last_suff_home_goal = suff_home_goal
                last_victories_hometeam_home = victories_hometeam_home
                last_losses_hometeam_home = losses_hometeam_home
                last_draws_hometeam_home = draws_hometeam_home

                if index < 37:

                    # descobrindo qual o proximo jogo
                    hometeam_next_game = df_temp.loc[index + 1, 'HomeTeam']
                    awayteam_next_game = df_temp.loc[index + 1, 'AwayTeam']

                    if hometeam_next_game == team:

                        temp_index = df.index[(df.HomeTeam == hometeam_next_game) & (df.AwayTeam == awayteam_next_game)]
                        df.loc[temp_index, 'HHGR'] = home_goal_ratio
                        df.loc[temp_index, 'HSHGR'] = suff_home_goal_ratio
                        df.loc[temp_index, 'VHHR'] = victories_hometeam_home_ratio
                        df.loc[temp_index, 'DHHR'] = draws_hometeam_home_ratio
                        df.loc[temp_index, 'LHHR'] = losses_hometeam_home_ratio

                    elif awayteam_next_game == team:

                        temp_index = df.index[(df.HomeTeam == hometeam_next_game) & (df.AwayTeam == awayteam_next_game)]
                        df.loc[temp_index, 'AAGR'] = away_goal_ratio
                        df.loc[temp_index, 'ASAGR'] = suff_away_goal_ratio
                        df.loc[temp_index, 'VAAR'] = victories_awayteam_away_ratio
                        df.loc[temp_index, 'DAAR'] = draws_awayteam_away_ratio
                        df.loc[temp_index, 'LAAR'] = losses_awayteam_away_ratio

            elif awayteam == team:

                if index == 0:

                    away_goal = df_temp.loc[index, 'FTAG'].item()
                    suff_away_goal = df_temp.loc[index, 'FTHG'].item()
                    away_goal_ratio = away_goal/counter_away_matches
                    suff_away_goal_ratio = suff_away_goal/counter_away_matches

                    match_result = df_temp.loc[index, 'FTR'].item()

                    if match_result == 1:

                        losses_awayteam_away = 1
                        losses_awayteam_away_ratio = 1

                    elif match_result == 2:

                        victories_awayteam_away = 1
                        victories_awayteam_away_ratio = 1

                    elif match_result == 0:

                        draws_awayteam_away = 1
                        draws_awayteam_away_ratio = 1


                elif index > 0:

                    away_goal = last_away_goal + df_temp.loc[index, 'FTAG'].item()
                    suff_away_goal = last_suff_away_goal + df_temp.loc[index, 'FTHG'].item()
                    away_goal_ratio = away_goal/counter_away_matches
                    suff_away_goal_ratio = suff_away_goal/counter_away_matches

                    match_result = df_temp.loc[index, 'FTR'].item()

                    if match_result == 1:

                        losses_awayteam_away = last_losses_awayteam_away + 1
                        losses_awayteam_away_ratio = losses_awayteam_away/counter_away_matches
                        victories_awayteam_away_ratio = victories_awayteam_away/counter_away_matches
                        draws_awayteam_away_ratio = draws_awayteam_away/counter_away_matches


                    elif match_result == 2:

                        victories_awayteam_away = last_victories_awayteam_away + 1
                        victories_awayteam_away_ratio = victories_awayteam_away/counter_away_matches
                        losses_awayteam_away_ratio = losses_awayteam_away/counter_away_matches
                        draws_awayteam_away_ratio = draws_awayteam_away/counter_away_matches

                    elif match_result == 0:

                        draws_awayteam_away = last_draws_awayteam_away + 1
                        draws_awayteam_away_ratio = draws_awayteam_away/counter_away_matches
                        victories_awayteam_away_ratio = victories_awayteam_away/counter_away_matches
                        losses_awayteam_away_ratio = losses_awayteam_away/counter_away_matches


                counter_away_matches += 1
                last_away_goal = away_goal
                last_suff_away_goal = suff_away_goal
                last_victories_awayteam_away = victories_awayteam_away
                last_losses_awayteam_away = losses_awayteam_away
                last_draws_awayteam_away = draws_awayteam_away

                # ultima partida de cada time nao precisa ser atualizada
                if index < 37:

                    # descobrindo qual o proximo jogo
                    hometeam_next_game = df_temp.loc[index + 1, 'HomeTeam']
                    awayteam_next_game = df_temp.loc[index + 1, 'AwayTeam']

                    if awayteam_next_game == team:

                        temp_index = df.index[(df.HomeTeam == hometeam_next_game) & (df.AwayTeam == awayteam_next_game)]
                        df.loc[temp_index, 'AAGR'] = away_goal_ratio
                        df.loc[temp_index, 'ASAGR'] = suff_away_goal_ratio
                        df.loc[temp_index, 'VAAR'] = victories_awayteam_away_ratio
                        df.loc[temp_index, 'DAAR'] = draws_awayteam_away_ratio
                        df.loc[temp_index, 'LAAR'] = losses_awayteam_away_ratio

                    elif hometeam_next_game == team:

                        temp_index = df.index[(df.HomeTeam == hometeam_next_game) & (df.AwayTeam == awayteam_next_game)]
                        df.loc[temp_index, 'HHGR'] = home_goal_ratio
                        df.loc[temp_index, 'HSHGR'] = suff_home_goal_ratio
                        df.loc[temp_index, 'VHHR'] = victories_hometeam_home_ratio
                        df.loc[temp_index, 'DHHR'] = draws_hometeam_home_ratio
                        df.loc[temp_index, 'LHHR'] = losses_hometeam_home_ratio

def calculate_poisson_dist(df):

    # todos os times ja possuem as razoes de gols calculadas em cada rodadas
    df_temp = df.copy()
    df_temp = df_temp.reset_index()

    df['AHG'] = 0
    df['AAG'] = 0

    # a cada rodada completa (10 jogos - 0 a 9 para o contador) da temporada
    HG = 0     # home goals
    AG = 0     # away goals
    AHG = 0    # average home goals
    AAG = 0    # average away goals

    for index, row in df_temp.iterrows():

        # contabilizando o total de gols dos times mandantes na rodada
        HG += df_temp.loc[index, 'FTHG'].item()
        AG += df_temp.loc[index, 'FTAG'].item()

        # dividindo o total de gols contabilizado pelo total de partidas
        AHG = HG/(index+1)
        AAG = AG/(index+1)

        df.loc[index,'AHG'] = AHG
        df.loc[index,'AAG'] = AAG

    df_temp = df.copy()
    df_temp = df_temp.reset_index()

    # inicializando as colunas
    df['HAS'] = 0
    df['AAS'] = 0
    df['HDS'] = 0
    df['ADS'] = 0
    df['HE'] = 0
    df['AE'] = 0

    # inicializando variáveis
    AAS = 0    # força do ataque de um time visitante
    HAS = 0    # força do ataque do time mandantes
    ADS = 0    # força da defesa de um time visitante
    HDS = 0    # força da defesa do time mandantes
    AHG = 0    # media de gols dos times mandantes até agora
    AAG = 0    # media de gols dos times visitantes ate agora
    HE = 0     # probabilidade do time mandante vencer
    AE = 0     # probabilidade do time visitante vencer
    draw_prob = 0
    home_win_prob = 0
    away_win_prob = 0
    h = 0
    a = 0
    df['D'] = 0
    df['HW'] = 0
    df['AW'] = 0

    for index, row in df_temp.iterrows():

        if index > 9:

            AHG = df.loc[index,'AHG'].item()
            AAG = df.loc[index,'AAG'].item()
            HAS = df.loc[index,'HHGR'].item()/AHG
            AAS = df.loc[index,'AAGR'].item()/AAG
            HDS = df.loc[index,'HSHGR'].item()/df.loc[index,'AAG'].item()
            ADS = df.loc[index,'ASAGR'].item()/df.loc[index,'AHG'].item()
            df.loc[index,'HAS'] = HAS
            df.loc[index,'AAS'] = AAS
            df.loc[index,'HDS'] = HDS
            df.loc[index,'ADS'] = ADS

            # calculando home expectancy e away expectancy
            HE = AHG*HAS*ADS
            AE = AAG*AAS*HDS
            df.loc[index,'HE'] = HE
            df.loc[index,'AE'] = AE

            H = df.loc[index,'HE'].item()
            A = df.loc[index,'AE'].item()
            # calculando chances de empate
            for x in range(0,10):

                for y in range(0,10):


                    h = poisson_probability(x, H)
                    a = poisson_probability(y, A)

                    if x == y:

                        draw_prob += (h*a)

                    elif x>y:

                        home_win_prob += (h*a)

                    elif x<y:

                        away_win_prob += (h*a)


            df.loc[index,'D'] = draw_prob
            df.loc[index,'HW'] = home_win_prob
            df.loc[index,'AW'] = away_win_prob
            total_prob = draw_prob + home_win_prob + away_win_prob

            if total_prob > 1.05 and df.loc[index,'Used'] == 1:

                print('!!!deu ruim!!!')
                print(df.loc[index,'HomeTeam'])
                print(df.loc[index,'AwayTeam'])

            draw_prob = 0
            home_win_prob = 0
            away_win_prob = 0

def poisson_probability(actual, mean):


    p = math.exp(-mean)*(mean**actual)/math.factorial(actual)

    return p

def main():

    global df_17

    print('-----------------------------------------')
    print('df_1')
    # calculando o ELO
    calculate_elo(df_1)
    # calculando features
    calculate_past_games_features(df_1)
    # calculando forma
    calculate_form(df_1)
    # calculando Poisson
    calculate_poisson_dist(df_1)
    #df_1.to_csv('df_1.csv', sep=';')
    # substituindo os times rebaixados pelos promovidos
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    update_teams(eigheenth='Man City', nineteenth='Coventry',
    twentieth = 'Bradford', first = 'Fulham', second = 'Blackburn',
    third = 'Bolton')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_2')
    # calculando o ELO
    calculate_elo(df_2)
    # calculando features
    calculate_past_games_features(df_2)
    calculate_form(df_2)
    # calculando Poisson
    calculate_poisson_dist(df_2)
    #df_2.to_csv('df_2.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Ipswich', nineteenth='Derby',
    twentieth = 'Leicester', first = 'Man City', second = 'West Brom',
    third = 'Birmingham')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_3')
    # calculando o ELO
    calculate_elo(df_3)
    # calculando features
    calculate_past_games_features(df_3)
    calculate_form(df_3)
    # calculando Poisson
    calculate_poisson_dist(df_3)
    #df_3.to_csv('df_3.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='West Ham', nineteenth='West Brom',
    twentieth = 'Sunderland', first = 'Portsmouth', second = 'Leicester',
    third = 'Wolves')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_4')
    # calculando o ELO
    calculate_elo(df_4)
    # calculando features
    calculate_past_games_features(df_4)
    calculate_form(df_4)
    # calculando Poisson
    calculate_poisson_dist(df_4)
    #df_4.to_csv('df_4.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Leicester', nineteenth='Leeds',
    twentieth = 'Wolves', first = 'Norwich', second = 'West Brom',
    third = 'Crystal Palace')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_5')
    # calculando o ELO
    calculate_elo(df_5)
    # calculando features
    calculate_past_games_features(df_5)
    calculate_form(df_5)
    # calculando Poisson
    calculate_poisson_dist(df_5)
    #df_5.to_csv('df_5.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Crystal Palace', nineteenth='Norwich',
    twentieth = 'Southampton', first = 'Sunderland', second = 'Wigan',
    third = 'West Ham')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_6')
    # calculando o ELO
    calculate_elo(df_6)
    # calculando features
    calculate_past_games_features(df_6)
    calculate_form(df_6)
    # calculando Poisson
    calculate_poisson_dist(df_6)
    #df_6.to_csv('df_6.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Birmingham', nineteenth='West Brom',
    twentieth = 'Sunderland', first = 'Reading', second = 'Sheffield United',
    third = 'Watford')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_7')
    # calculando o ELO
    calculate_elo(df_7)
    # calculando features
    calculate_past_games_features(df_7)
    calculate_form(df_7)
    # calculando Poisson
    calculate_poisson_dist(df_7)
    #df_7.to_csv('df_7.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Sheffield United', nineteenth='Charlton',
    twentieth = 'Watford', first = 'Sunderland', second = 'Birmingham',
    third = 'Derby')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_8')
    # calculando o ELO
    calculate_elo(df_8)
    # calculando features
    calculate_past_games_features(df_8)
    calculate_form(df_8)
    # calculando Poisson
    calculate_poisson_dist(df_8)
    #df_8.to_csv('df_8.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Reading', nineteenth='Birmingham',
    twentieth = 'Derby', first = 'West Brom', second = 'Stoke',
    third = 'Hull')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_9')
    # calculando o ELO
    calculate_elo(df_9)
    # calculando features
    calculate_past_games_features(df_9)
    calculate_form(df_9)
    # calculando Poisson
    calculate_poisson_dist(df_9)
    #df_9.to_csv('df_9.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Newcastle', nineteenth='Middlesbrough',
    twentieth = 'West Brom', first = 'Wolves', second = 'Birmingham',
    third = 'Burnley')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_10')
    # calculando o ELO
    calculate_elo(df_10)
    # calculando features
    calculate_past_games_features(df_10)
    calculate_form(df_10)
    # calculando Poisson
    calculate_poisson_dist(df_10)
    #df_10.to_csv('df_10.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Burnley', nineteenth='Hull',
    twentieth = 'Portsmouth', first = 'Newcastle', second = 'West Brom',
    third = 'Blackpool')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_11')
    # calculando o ELO
    calculate_elo(df_11)
    # calculando features
    calculate_past_games_features(df_11)
    calculate_form(df_11)
    # calculando Poisson
    calculate_poisson_dist(df_11)
    #df_11.to_csv('df_11.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Birmingham', nineteenth='Blackpool',
    twentieth = 'West Ham', first = 'QPR', second = 'Norwich',
    third = 'Swansea')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_12')
    # calculando o ELO
    calculate_elo(df_12)
    # calculando features
    calculate_past_games_features(df_12)
    calculate_form(df_12)
    # calculando Poisson
    calculate_poisson_dist(df_12)
    #df_12.to_csv('df_12.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Bolton', nineteenth='Blackburn',
    twentieth = 'Wolves', first = 'Reading', second = 'Southampton',
    third = 'West Ham')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_13')
    # calculando o ELO
    calculate_elo(df_13)
    # calculando features
    calculate_past_games_features(df_13)
    calculate_form(df_13)
    # calculando Poisson
    calculate_poisson_dist(df_13)
    #df_13.to_csv('df_13.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Wigan', nineteenth='Reading',
    twentieth = 'QPR', first = 'Cardiff', second = 'Hull',
    third = 'Crystal Palace')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_14')
    # calculando o ELO
    calculate_elo(df_14)
    # calculando features
    calculate_past_games_features(df_14)
    calculate_form(df_14)
    # calculando Poisson
    calculate_poisson_dist(df_14)
    #df_14.to_csv('df_14.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Norwich', nineteenth='Fulham',
    twentieth = 'Cardiff', first = 'Leicester', second = 'Burnley',
    third = 'QPR')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_15')
    # calculando o ELO
    calculate_elo(df_15)
    # calculando features
    calculate_past_games_features(df_15)
    calculate_form(df_15)
    # calculando Poisson
    calculate_poisson_dist(df_15)
    #df_15.to_csv('df_15.csv', sep=';')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Hull', nineteenth='Burnley',
    twentieth = 'QPR', first = 'Bournemouth', second = 'Watford',
    third = 'Norwich')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_16')
    # calculando o ELO
    calculate_elo(df_16)
    # calculando features
    calculate_past_games_features(df_16)
    calculate_form(df_16)
    # calculando Poisson
    calculate_poisson_dist(df_16)
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    # substituindo os times rebaixados pelos promovidos
    update_teams(eigheenth='Newcastle', nineteenth='Norwich',
    twentieth = 'Aston Villa', first = 'Burnley', second = 'Middlesbrough',
    third = 'Hull')
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    print('-----------------------------------------')
    print('df_17')
    # calculando o ELO
    calculate_elo(df_17)
    # calculando features
    calculate_past_games_features(df_17)
    calculate_form(df_17)
    # calculando Poisson
    calculate_poisson_dist(df_17)
    total=0
    for index, row in df_unique.iterrows():
        total += df_unique.loc[index,'ELO'].item()
    print('total')
    print(total)
    print(df_unique)
    print('-----------------------------------------')

    dfs = [df_3,df_4,df_5,df_6,df_7,df_8,df_9,df_10,df_11,df_12,df_13,df_14,df_15]
    df = pd.concat(dfs)
    df.to_csv('df_england_3-15_10out.csv', sep=';')
    print('df gerado com sucesso')

    dfs2 = [df_16,df_17]
    df2 = pd.concat(dfs2)
    df2.to_csv('df_england_16-17_10out.csv', sep=';')
    print('df gerado com sucesso')

if __name__ == '__main__':
    main()
