import pandas as pd
import numpy as np
# carregando para gerar gráficos
from matplotlib import pyplot as pl
from matplotlib import pylab
import matplotlib.lines as mlines



acc_19_2 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_2.csv", sep=';')
df_acc_19_it_2 = pd.DataFrame(acc_19_2, columns = ['0','1','2'])

acc_19_3 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_3.csv", sep=';')
df_acc_19_it_3 = pd.DataFrame(acc_19_3, columns = ['0','1','2'])

acc_19_4 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_4.csv", sep=';')
df_acc_19_it_4 = pd.DataFrame(acc_19_4, columns = ['0','1','2'])

acc_19_5 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_5.csv", sep=';')
df_acc_19_it_5 = pd.DataFrame(acc_19_5, columns = ['0','1','2'])

acc_19_6 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_6.csv", sep=';')
df_acc_19_it_6 = pd.DataFrame(acc_19_6, columns = ['0','1','2'])

acc_19_7 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_7.csv", sep=';')
df_acc_19_it_7 = pd.DataFrame(acc_19_7, columns = ['0','1','2'])

acc_19_8 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_8.csv", sep=';')
df_acc_19_it_8 = pd.DataFrame(acc_19_8, columns = ['0','1','2'])

acc_19_9 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_9.csv", sep=';')
df_acc_19_it_9 = pd.DataFrame(acc_19_9, columns = ['0','1','2'])

acc_19_10 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_10.csv", sep=';')
df_acc_19_it_10 = pd.DataFrame(acc_19_10, columns = ['0','1','2'])

acc_19_11 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_11.csv", sep=';')
df_acc_19_it_11 = pd.DataFrame(acc_19_11, columns = ['0','1','2'])

acc_19_12 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_12.csv", sep=';')
df_acc_19_it_12 = pd.DataFrame(acc_19_12, columns = ['0','1','2'])

acc_19_13 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_13.csv", sep=';')
df_acc_19_it_13 = pd.DataFrame(acc_19_13, columns = ['0','1','2'])

acc_19_14 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_14.csv", sep=';')
df_acc_19_it_14 = pd.DataFrame(acc_19_14, columns = ['0','1','2'])

acc_19_15 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_15.csv", sep=';')
df_acc_19_it_15 = pd.DataFrame(acc_19_15, columns = ['0','1','2'])

acc_19_16 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_16.csv", sep=';')
df_acc_19_it_16 = pd.DataFrame(acc_19_16, columns = ['0','1','2'])

acc_19_17 = pd.read_csv("../plot_feature_selection/\
df_acc_19_it_17.csv", sep=';')
df_acc_19_it_17 = pd.DataFrame(acc_19_17, columns = ['0','1','2'])


dfs = [df_acc_19_it_2,df_acc_19_it_3,df_acc_19_it_4,df_acc_19_it_5,df_acc_19_it_6,df_acc_19_it_7,
df_acc_19_it_8,df_acc_19_it_9,df_acc_19_it_10,df_acc_19_it_11,df_acc_19_it_12,df_acc_19_it_13,
df_acc_19_it_14,df_acc_19_it_15,df_acc_19_it_16,df_acc_19_it_17]

df = pd.concat(dfs)

df.columns = ['Nome','Media','DP']

df_RL = df.loc[df.Nome == 'RL']

RLnome = df_RL['Nome'].tolist()
RLmedia = df_RL['Media'].tolist()
RLDP = df_RL['DP'].tolist()

x = np.linspace(2,17,16)
yRL = RLmedia
errorRL = RLDP
ymerrorRL = [yRL - errorRL for yRL, errorRL in zip(yRL, errorRL)]
yperrorRL = [yRL + errorRL for yRL, errorRL in zip(yRL, errorRL)]


fig = pl.figure()
# RL
pl.subplot(511)
pl.ylim([35,60])
pl.xlim([2,17])
pl.yticks(rotation='horizontal')
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
pl.plot(x, yRL, 'k', color='#3F7F4C')
pl.fill_between(x, ymerrorRL, yperrorRL,
                alpha=0.5, edgecolor='#3F7F4C', facecolor='#7EFF99')

df_ADL = df.loc[df.Nome == 'ADL']

ADLnome = df_ADL['Nome'].tolist()
ADLmedia = df_ADL['Media'].tolist()
ADLDP = df_ADL['DP'].tolist()

x = np.linspace(2,17,16)
yADL = ADLmedia
errorADL = ADLDP
ymerrorADL = [yADL - errorADL for yADL, errorADL in zip(yADL, errorADL)]
yperrorADL = [yADL + errorADL for yADL, errorADL in zip(yADL, errorADL)]

# ADL
pl.subplot(512)
pl.ylim([35,60])
pl.xlim([2,17])
pl.yticks(rotation='horizontal')
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
pl.plot(x, yADL, 'k', color='#1B2ACC')
pl.fill_between(x, ymerrorADL, yperrorADL,
                alpha=0.5, edgecolor='#1B2ACC', facecolor='#ABEBC6')


df_ADQ = df.loc[df.Nome == 'ADQ']

ADQnome = df_ADQ['Nome'].tolist()
ADQmedia = df_ADQ['Media'].tolist()
ADQDP = df_ADQ['DP'].tolist()

x = np.linspace(2,17,16)
yADQ = ADQmedia
errorADQ = ADQDP
ymerrorADQ = [yADQ - errorADQ for yADQ, errorADQ in zip(yADQ, errorADQ)]
yperrorADQ = [yADQ + errorADQ for yADQ, errorADQ in zip(yADQ, errorADQ)]

pl.subplot(513)
pl.ylim([35,60])
pl.xlim([2,17])
pl.yticks(rotation='horizontal')
pl.ylabel('Acurácia', fontsize = 16)
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
pl.plot(x, yADQ, 'k', color='#6C3483')
pl.fill_between(x, ymerrorADQ, yperrorADQ,
                alpha=0.5, edgecolor='#6C3483', facecolor='#BB8FCE')

df_KNN = df.loc[df.Nome == 'KNN']

KNNnome = df_KNN['Nome'].tolist()
KNNmedia = df_KNN['Media'].tolist()
KNNDP = df_KNN['DP'].tolist()

x = np.linspace(2,17,16)
yKNN = KNNmedia
errorKNN = KNNDP
ymerrorKNN = [yKNN - errorKNN for yKNN, errorKNN in zip(yKNN, errorKNN)]
yperrorKNN = [yKNN + errorKNN for yKNN, errorKNN in zip(yKNN, errorKNN)]

pl.subplot(514)
pl.ylim([35,60])
pl.xlim([2,17])
pl.yticks(rotation='horizontal')
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
pl.plot(x, yKNN, 'k', color='#FF8810')
pl.fill_between(x, ymerrorKNN, yperrorKNN,
                alpha=0.5, edgecolor='#FF8810', facecolor='#F5D8BA')

df_NBG = df.loc[df.Nome == 'NBG']

NBGnome = df_NBG['Nome'].tolist()
NBGmedia = df_NBG['Media'].tolist()
NBGDP = df_NBG['DP'].tolist()

x = np.linspace(2,17,16)
yNBG = NBGmedia
errorNBG = NBGDP
ymerrorNBG = [yNBG - errorNBG for yNBG, errorNBG in zip(yNBG, errorNBG)]
yperrorNBG = [yNBG + errorNBG for yNBG, errorNBG in zip(yNBG, errorNBG)]

pl.subplot(514)
pl.ylim([35,60])
pl.xlim([2,17])
pl.yticks(rotation='horizontal')
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
pl.plot(x, yNBG, 'k', color='#17202A')
pl.fill_between(x, ymerrorNBG, yperrorNBG,
                alpha=0.5, edgecolor='#17202A', facecolor='#85929E')


df_NBM = df.loc[df.Nome == 'NBM']

NBMnome = df_NBM['Nome'].tolist()
NBMmedia = df_NBM['Media'].tolist()
NBMDP = df_NBM['DP'].tolist()

x = np.linspace(2,17,16)
yNBM = NBMmedia
errorNBM = NBMDP
ymerrorNBM = [yNBM - errorNBM for yNBM, errorNBM in zip(yNBM, errorNBM)]
yperrorNBM = [yNBM + errorNBM for yNBM, errorNBM in zip(yNBM, errorNBM)]

pl.subplot(511)
pl.ylim([35,60])
pl.xlim([2,17])
pl.yticks(rotation='horizontal')
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
pl.plot(x, yNBM, 'k', color='#CB4335')
pl.fill_between(x, ymerrorNBM, yperrorNBM,
                alpha=0.5, edgecolor='#CB4335', facecolor='#F1948A')

df_SVML = df.loc[df.Nome == 'SVML']

SVMLnome = df_SVML['Nome'].tolist()
SVMLmedia = df_SVML['Media'].tolist()
SVMLDP = df_SVML['DP'].tolist()

x = np.linspace(2,17,16)
ySVML = SVMLmedia
errorSVML = SVMLDP
ymerrorSVML = [ySVML - errorSVML for ySVML, errorSVML in zip(ySVML, errorSVML)]
yperrorSVML = [ySVML + errorSVML for ySVML, errorSVML in zip(ySVML, errorSVML)]

pl.subplot(513)
pl.ylim([35,60])
pl.xlim([2,17])
pl.yticks(rotation='horizontal')
pl.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off')
pl.plot(x, ySVML, 'k', color='#641E16')
pl.fill_between(x, ymerrorSVML, yperrorSVML,
                alpha=0.5, edgecolor='#641E16', facecolor='#922B21')

df_SVMR = df.loc[df.Nome == 'SVMR']

SVMRnome = df_SVMR['Nome'].tolist()
SVMRmedia = df_SVMR['Media'].tolist()
SVMRDP = df_SVMR['DP'].tolist()

x = np.linspace(2,17,16)
ySVMR = SVMRmedia
errorSVMR = SVMRDP
ymerrorSVMR = [ySVMR - errorSVMR for ySVMR, errorSVMR in zip(ySVMR, errorSVMR)]
yperrorSVMR = [ySVMR + errorSVMR for ySVMR, errorSVMR in zip(ySVMR, errorSVMR)]

pl.subplot(515)
pl.xlabel('Quantidade de Features', fontsize = 16)
pl.ylim([35,60])
pl.xlim([2,17])
pl.xticks(rotation='horizontal')
pl.yticks(rotation='horizontal')
pl.plot(x, ySVMR, 'k', color='#FD01E0')
pl.fill_between(x, ymerrorSVMR, yperrorSVMR,
                alpha=0.5, edgecolor='#FD01E0', facecolor='#F8E1F6')

df_RF = df.loc[df.Nome == 'RF']

RFnome = df_RF['Nome'].tolist()
RFmedia = df_RF['Media'].tolist()
RFDP = df_RF['DP'].tolist()

x = np.linspace(2,17,16)
yRF = RFmedia
errorRF = RFDP
ymerrorRF = [yRF - errorRF for yRF, errorRF in zip(yRF, errorRF)]
yperrorRF = [yRF + errorRF for yRF, errorRF in zip(yRF, errorRF)]

pl.subplot(512)
pl.ylim([35,60])
pl.xlim([2,17])
pl.xticks(rotation='horizontal')
pl.yticks(rotation='horizontal')
pl.plot(x, yRF, 'k', color='#27FF0B')
pl.fill_between(x, ymerrorRF, yperrorRF,
                alpha=0.5, edgecolor='#27FF0B', facecolor='#D1F4CD')

df_ET = df.loc[df.Nome == 'ET']

ETnome = df_ET['Nome'].tolist()
ETmedia = df_ET['Media'].tolist()
ETDP = df_ET['DP'].tolist()

x = np.linspace(2,17,16)
yET = ETmedia
errorET = ETDP
ymerrorET = [yET - errorET for yET, errorET in zip(yET, errorET)]
yperrorET = [yET + errorET for yET, errorET in zip(yET, errorET)]

pl.subplot(515)
pl.ylim([35,60])
pl.xlim([2,17])
pl.xticks(rotation='horizontal')
pl.yticks(rotation='horizontal')
pl.plot(x, yET, 'k', color='#b8860b')
pl.fill_between(x, ymerrorET, yperrorET,
                alpha=0.5, edgecolor='#b8860b', facecolor='#eedd82')


#pl.gca().set_color_cycle(['#3F7F4C', '#1B2ACC', '#6C3483','#b8860b',
#                          '#17202A','#CB4335', '#641E16', '#FD01E0',
#                          '#27FF0B','#FF8810'])
#pl.legend(['RL', 'ADL', 'ADQ', 'KNN', 'NBG', 'NBM', 'SVML', 'SVMR',
#           'RF', 'ET' ], loc='lower right',bbox_to_anchor=(1.2, 2.1))

# Create custom legend
line_1 = mlines.Line2D([], [], color='#3F7F4C',markersize=15, label='RL')
line_2 = mlines.Line2D([], [], color='#1B2ACC', markersize=15, label='ADL')
line_3 = mlines.Line2D([], [], color='#6C3483',markersize=15, label='ADQ')
line_4 = mlines.Line2D([], [], color='#FF8810', markersize=15, label='KNN')
line_5 = mlines.Line2D([], [], color='#17202A',markersize=15, label='NBG')
line_6 = mlines.Line2D([], [], color='#CB4335', markersize=15, label='NBM')
line_7 = mlines.Line2D([], [], color='#641E16',markersize=15, label='SVML')
line_8 = mlines.Line2D([], [], color='#FD01E0', markersize=15, label='SVMR')
line_9 = mlines.Line2D([], [], color='#27FF0B',markersize=15, label='RF')
line_10 = mlines.Line2D([], [], color='#b8860b', markersize=15, label='ET')

pl.legend(handles=[line_1,line_2,line_3,line_4,line_5,line_6,line_7,line_8,line_9,line_10],
                   loc='lower right',bbox_to_anchor=(1.2, 1.2))

pl.show()
