import pandas as pd
# carregando para gerar gráficos
import matplotlib.pyplot as plt
import seaborn as sns
# carregando numpy para ações matemáticas
import numpy as np
import itertools
import scipy
from scipy import stats
# carregando para gerar matriz de confusão
from sklearn.metrics import confusion_matrix
# validação e métricas
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn import model_selection
from sklearn.metrics import classification_report
# classificadores
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import VotingClassifier


# variáveis globais

names_plots = ['EM','EV','PVM','PVV','PE','MGM','MGV',
	           'MGSM','MGSV','MEM','MEV','MVM','MVV','FM','FV','EGM','EGV','R']

names_plots_no_r = ['EM','EV','PVM','PVV','PE','MGM','MGV',
	                'MGSM','MGSV','MEM','MEV','MVM','MVV','FM','FV','EGM','EGV']

# variável que seta a quantidade de partidas ignoradas em cada temporada
part_ign = 19


# funcao que plota a matriz de correlacao das features
def correlation_features(df):

	# analisando a correlação das features
    df_corr = df.copy()

    # Used não interessa nesse momento
    del df_corr['Used']

    pearson = df_corr.corr()
    pearson.to_csv('correlations.csv', sep=';')

    # analise de algumas caracteristicas da correlacao entre variaveis
    corr_with_target = pearson.ix[-1][:-1]
    corr_with_target = (corr_with_target[abs(corr_with_target).argsort()[::-1]])
    corr_with_target.to_csv('corr_with_target.csv', sep=';')

    attrs = pearson.iloc[:-1,:-1] # todas menos o Resultado

    # apenas correlações acima de um certo threshold
    threshold = 0.7
    important_corrs = (attrs[abs(attrs) > threshold][attrs != 1.0]) \
        .unstack().dropna().to_dict()

    unique_important_corrs = pd.DataFrame(
        list(set([(tuple(sorted(key)), important_corrs[key]) \
        for key in important_corrs])), columns=['attribute pair', 'correlation'])
    # classificando por valor absoluto
    unique_important_corrs = unique_important_corrs.ix[abs(unique_important_corrs['correlation']).argsort()[::-1]]
    unique_important_corrs.to_csv('unique_important_corrs.csv', sep=';')
    print(unique_important_corrs)

    # plotar matriz de correlação
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(pearson, vmin=-1, vmax=1)
    fig.colorbar(cax, fraction=0.046, pad=0.04, orientation = 'horizontal').set_label('Correlacao', fontsize=12)
    ticks = np.arange(0,22,1)
    ax.set_xticks(ticks)
    plt.xticks(rotation='vertical', fontsize=12)
    plt.yticks(rotation='horizontal', fontsize=12)
    ax.set_yticks(ticks)
    ax.set_xticklabels(names_plots)
    ax.set_yticklabels(names_plots)

	# legenda das features
    ax.annotate('EM = ELO Mandante\nEV = ELO Visitante\n\
PVM = Poisson Vitoria Mandante\nPVV = Poisson Vitoria Visitante\n\
PE = Poisson Empate\n\
MGM = Media Gols Mandante\n\
MGV = Media Gols Visitante\n\
MGSM = Media Gols Sofridos Mandante\n\
MGSV = Media Gols Sofridos Visitante\n\
MEM = Media Empates Mandante\n\
MEV = Media Empates Visitante\n\
MVM = Media Vitorias Mandante\n\
MVV = Media Vitorias Visitante\n\
FM = Forma Mandante\n\
FV = Forma Visitante\n\
EGM = Expectativa Gols Mandante\n\
EGV = Expectativa Gols Visitante\n\
R = Resultado',
				xy=(1.05, 0.5),
				xycoords=('axes fraction', 'figure fraction'),
				xytext=(0, 0),
				textcoords='offset points',
				size=12, ha='left', va='center')
    plt.show()


def plot_confusion_matrix(y_true, y_pred,
                             classes,
                             normalize=False,
                             title='Matriz de Confusao',
                             cmap=plt.cm.Blues):
    """
    Source: http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html#sphx-glr-auto-examples-model-selection-plot-confusion-matrix-py

    """

    cm = confusion_matrix(y_true, y_pred)

    # Configure Confusion Matrix Plot Aesthetics (no text yet)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=20)
    tick_marks = np.arange(len(classes))
    names_classes = ['Empate','Vitoria M','Vitoria V']
    plt.xticks(range(len(names_classes)), names_classes)
    plt.yticks(range(len(names_classes)), names_classes)
    plt.xticks(rotation='horizontal', fontsize=14)
    plt.yticks(rotation='horizontal', fontsize=14)
    plt.ylabel('Real', fontsize=18)
    plt.xlabel('Previsto', fontsize=18)

    # Calculate normalized values (so all cells sum to 1) if desired
    if normalize:
        cm = np.round(cm.astype('float') / cm.sum(),2) #(axis=1)[:, np.newaxis]

    # Place Numbers as Text on Confusion Matrix Plot
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black",
                 fontsize=18)



    plt.show()

def k_fold_cross_validation(x, y, splits, repeats):

    seed = 7

    # classificadores para o ensemble
    clf1 = LogisticRegression(random_state=seed,C=625, penalty='l1')
    clf2 = MultinomialNB(alpha=1130)
    clf3 = GaussianNB()
    clf4 = KNeighborsClassifier(n_neighbors=450)
    clf5 = ExtraTreesClassifier(random_state = seed,criterion='gini',
                                n_estimators=1000,max_features=5)
    clf6 = QuadraticDiscriminantAnalysis()
    eclf = VotingClassifier(estimators=[('LR', clf1), ('NBM', clf2), ('NBG', clf3), ('KNN', clf4), ('ET', clf5), ('ADQ', clf6)], voting='hard')

    # Algoritmos comparados
    models = []

    models.append(('RL', LogisticRegression(random_state=seed,
                                            C=625, penalty='l1')))
    models.append(('ADL', LinearDiscriminantAnalysis()))
    models.append(('ADQ', QuadraticDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=450)))
    models.append(('NBG', GaussianNB()))
    models.append(('NBM', MultinomialNB(alpha=1130)))
    models.append(('SVML', SVC(random_state=seed, kernel='linear', C=0.1)))
    models.append(('SVMR', SVC(random_state=seed,
                               kernel='rbf', C=1, gamma=0.0001)))
    models.append(('RF', RandomForestClassifier(random_state = seed,
                                                criterion='entropy',
                                                n_estimators=1000,
                                                max_features=5)))
    models.append(('ET', ExtraTreesClassifier(random_state = seed,
                                              criterion='gini',
                                              n_estimators=1000,
                                              max_features=5)))
    models.append(('ENS', eclf))

    # loop que analisa cada algoritmo
    score = 'accuracy'
    results1 = []
    names1 = []
    mean1 = []
    std1 = []

    for name, model in models:
        kfold = model_selection.RepeatedStratifiedKFold(n_splits=splits,
                                                        n_repeats = repeats,
                                                        random_state=seed)
        cv_results = model_selection.cross_val_score(model,x, y,
                                                     cv=kfold,
                                                     scoring=score)
        results1.append(cv_results)
        names1.append(name)
        mean1.append(cv_results.mean()*100)
        std1.append(cv_results.std()*100)
        msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
        print(msg)

    list_results_acc = list(zip(names1,results1))
    print(list_results_acc)
    df_results_acc = pd.DataFrame(list_results_acc)
    if part_ign == 3:
        df_results_acc.to_csv('df_results_acc_3.csv', sep=';')
    if part_ign == 10:
        df_results_acc.to_csv('df_results_acc_10.csv', sep=';')
    if part_ign == 19:
        df_results_acc.to_csv('df_results_acc_19.csv', sep=';')

    if score == 'accuracy':
        list_acc = list(zip(names1, mean1, std1))
        df_acc = pd.DataFrame(list_acc)
        if part_ign == 3:
            df_acc.to_csv('df_acc_3.csv', sep=';')
        if part_ign == 10:
            df_acc.to_csv('df_acc_10.csv', sep=';')
        if part_ign == 19:
            df_acc.to_csv('df_acc_19.csv', sep=';')


    # classificadores para o ensemble
    clf1 = LogisticRegression(random_state=seed,C=625, penalty='l1')
    clf2 = MultinomialNB(alpha=15)
    clf3 = GaussianNB()
    clf4 = KNeighborsClassifier(n_neighbors=10)
    clf5 = ExtraTreesClassifier(random_state = seed,criterion='entropy',
                                n_estimators=1000,max_features=17)
    clf6 = QuadraticDiscriminantAnalysis()
    eclf = VotingClassifier(estimators=[('LR', clf1), ('NBM', clf2), ('NBG', clf3), ('KNN', clf4), ('ET', clf5), ('ADQ', clf6)], voting='hard')

    models = []

    models.append(('RL', LogisticRegression(random_state=seed,
                                            C=625, penalty='l1')))
    models.append(('ADL', LinearDiscriminantAnalysis()))
    models.append(('ADQ', QuadraticDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=10)))
    models.append(('NBG', GaussianNB()))
    models.append(('NBM', MultinomialNB(alpha=15)))
    models.append(('SVML', SVC(random_state=seed, kernel='linear', C=10)))
    models.append(('SVMR', SVC(random_state=seed,
                               kernel='rbf', C=10, gamma=0.001)))
    models.append(('RF', RandomForestClassifier(random_state = seed,
                                                criterion='gini',
                                                n_estimators=1000,
                                                max_features=17)))
    models.append(('ET', ExtraTreesClassifier(random_state = seed,
                                              criterion='entropy',
                                              n_estimators=1000,
                                              max_features=17)))
    models.append(('ENS', eclf))

    # loop que analisa cada algoritmo
    score = 'f1_macro'
    results2 = []
    names2 = []
    mean2 = []
    std2 = []

    for name, model in models:
        kfold = model_selection.RepeatedStratifiedKFold(n_splits=splits,
                                                        n_repeats = repeats,
                                                        random_state=seed)
        cv_results = model_selection.cross_val_score(model,x, y,
                                                     cv=kfold,
                                                     scoring=score)
        results2.append(cv_results)
        names2.append(name)
        mean2.append(cv_results.mean()*100)
        std2.append(cv_results.std()*100)
        msg = "%s: %f (%f)" % (name, cv_results.mean()*100, cv_results.std()*100)
        print(msg)

    list_results_f1 = list(zip(names2,results2))
    print(list_results_f1)
    df_results_f1 = pd.DataFrame(list_results_f1)
    if part_ign == 3:
        df_results_f1.to_csv('df_results_f1_3.csv', sep=';')
    if part_ign == 10:
        df_results_f1.to_csv('df_results_f1_10.csv', sep=';')
    if part_ign == 19:
        df_results_f1.to_csv('df_results_f1_10.csv', sep=';')

    if score == 'f1_macro':
        list_f1 = list(zip(names2, mean2, std2))
        df_f1 = pd.DataFrame(list_f1)
        if part_ign == 3:
            df_f1.to_csv('df_f1_3.csv', sep=';')
        if part_ign == 10:
            df_f1.to_csv('df_f1_10.csv', sep=';')
        if part_ign == 19:
            df_f1.to_csv('df_f1_19.csv', sep=';')

	# plotando gráfico
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    plt.subplot(211)
    plt.boxplot(results1)
    ax1.set_xticklabels(names1,fontsize = 14)
    plt.ylabel('Acurácia', fontsize=18)
    plt.xlabel('(a)', fontsize=18)
    plt.yticks(rotation='horizontal', fontsize=14)
    plt.axhline(y=0.4656, xmin=0, xmax=1, hold=None, color='g')
    plt.axhline(y=0.5024, xmin=0, xmax=1, hold=None, color='b')
    plt.subplot(212)
    plt.xlabel('(b)\nClassificadores', fontsize=18)
    plt.boxplot(results2)
    plt.ylabel('F1-score', fontsize=18)
    ax2.set_xticklabels(names2,fontsize = 14)
    plt.yticks(rotation='horizontal', fontsize=14)
    ax2.annotate('RL = Regressao Logistica\nADL = Analise Discr. Linear\n\
ADQ = Analise Discr. Quadratica\nKNN = K-Nearest Neighbors\n\
NBG = Naive Bayes Gaussiano\nNBM = Naive Bayes Multinomial\n\
SVML = SVM Linear\nSVMR = SVM kernel rbf\nRF = Random Forest\n\
ET = Extra Trees',

				# The point that we'll place the text in relation to
				xy=(1.01, 0.5),
				# Interpret the x as axes coords, and the y as figure coords
				xycoords=('axes fraction', 'figure fraction'),

				# The distance from the point that the text will be at
				xytext=(0, 0),
				# Interpret `xytext` as an offset in points...
				textcoords='offset points',

				# Any other text parameters we'd like
				size=12, ha='left', va='center')
    plt.subplot(212)
    plt.show(fig)

def variable_importance_RF(estimators,x,y, criteria):


    print('Quantidade de partidas utilizadas na analise de importancia das variaveis:', len(x))

    # previsao de resultados
    clf = RandomForestClassifier(n_estimators=estimators, criterion=criteria)
    clf.fit(x, y)

    importance_plot = clf.feature_importances_
    importance_plot = pd.DataFrame(importance_plot, index=x.columns,
                                   columns=['Importance'])

    importance_plot['Std'] = np.std([tree.feature_importances_
                                for tree in clf.estimators_], axis=0)

    x = range(importance_plot.shape[0])
    y = importance_plot.ix[:, 0]
    yerr = importance_plot.ix[:, 1]

    fig = plt.figure()
    ax = fig.add_subplot(111)
    # legenda das features
    ax.annotate('EM = ELO Mandante\nEV = ELO Visitante\n\
PVM = Poisson Vitoria Mandante\nPVV = Poisson Vitoria Visitante\n\
PE = Poisson Empate\n\
MGM = Media Gols Mandante\n\
MGV = Media Gols Visitante\n\
MGSM = Media Gols Sofridos Mandante\n\
MGSV = Media Gols Sofridos Visitante\n\
MEM = Media Empates Mandante\n\
MEV = Media Empates Visitante\n\
MVM = Media Vitorias Mandante\n\
MVV = Media Vitorias Visitante\n\
FM = Forma Mandante\n\
FV = Forma Visitante\n\
EGM = Expectativa Gols Mandante\n\
EGV = Expectativa Gols Visitante',

				# posicionamento das legendas das features
				xy=(1.05, 0.5),
				xycoords=('axes fraction', 'figure fraction'),
				xytext=(0, 0),
				textcoords='offset points',
				size=12, ha='left', va='center')

    #plot
    plt.xticks(range(len(names_plots_no_r)), names_plots_no_r)
    plt.bar(x, y, yerr=yerr, align='center')
    plt.ylabel('Importância',fontsize=14)
    plt.xlabel('Feature',fontsize=14)
    plt.xticks(rotation='vertical', fontsize=12)
    plt.yticks(rotation='horizontal', fontsize=12)
    plt.show()


def gridSearch_classifier(x, y, tuned_parameters, scores, classifier):

    for score in scores:
        print("# Procurando o melhor hiperparametro com relacao a metrica %s" % score)
        print()

        clf = GridSearchCV(classifier, tuned_parameters, cv=10,scoring='%s' % score)
        clf.fit(x, y)

        print("Melhor hiperparametro encontrado:")
        print()
        print(clf.best_params_)
        print()
        print("Metricas alcancadas:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
        print()

def classification_report_csv(report, classifier):

    #fonte = https://stackoverflow.com/questions/39662398/scikit-learn-output-metrics-classification-report-into-csv-tab-delimited-format

    report_data = []
    lines = report.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv('classification_report_%s.csv' % classifier, index = False, sep=';')

def test_and_plot_CM(x_train,y_train,x_test,y_true,ign):

    if ign == 19:
        matches = 380

    if ign == 10:
        matches = 560

    if ign == 3:
        matches = 700

    seed = 7

    # classificadores para o ensemble
    clf1 = LogisticRegression(random_state=seed,C=625, penalty='l1')
    clf2 = MultinomialNB(alpha=1130)
    clf3 = GaussianNB()
    clf4 = KNeighborsClassifier(n_neighbors=450)
    clf5 = ExtraTreesClassifier(random_state = seed,criterion='gini',
                                n_estimators=1000,max_features=5)
    clf6 = QuadraticDiscriminantAnalysis()
    eclf = VotingClassifier(estimators=[('LR', clf1), ('NBM', clf2), ('NBG', clf3), ('KNN', clf4), ('ET', clf5), ('ADQ', clf6)], voting='hard')

    models = []
    names = []
    # demais classificadores
    models.append(('RL', LogisticRegression(random_state=seed,
                                            C=625, penalty='l1')))
    models.append(('ADL', LinearDiscriminantAnalysis()))
    models.append(('ADQ', QuadraticDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier(n_neighbors=450)))
    models.append(('NBG', GaussianNB()))
    models.append(('NBM', MultinomialNB(alpha=1130)))
    models.append(('SVML', SVC(random_state=seed, kernel='linear', C=0.1)))
    models.append(('SVMR', SVC(random_state=seed,
                               kernel='rbf', C=1, gamma=0.0001)))
    models.append(('RF', RandomForestClassifier(random_state = seed,
                                                criterion='entropy',
                                                n_estimators=1000,
                                                max_features=5)))
    models.append(('ET', ExtraTreesClassifier(random_state = seed,
                                              criterion='gini',
                                              n_estimators=1000,
                                              max_features=5)))
    models.append(('ENS', eclf))

    for name, model in models:

        # treinando classificador
        model.fit(x_train, y_train)
        # testando em dados novos
        pred = [model.predict(x_test)]
        df_pred = pd.DataFrame(pred)
        df_pred.to_csv('df_pred_%s.csv' % name, sep=';')
        pred = np.reshape(pred, (matches,1))

        report = classification_report(y_true, pred, target_names=['Empate','Vitória M','Vitória V'])
        classification_report_csv(report, name)

def scatter_plot():

    # valores que serão plotados
    # 3 partidas
    plt.subplot(311)
    points = [[52.40,38.23],[52.39,38.80],[43.60,38.15],[53.20,40.94],
              [49.22,41.49],[52.95,42.70],[52.85,38.83],[53.09,40.88],
              [51.42,41.09],[51.69,41.19]]
    names = ['RL','ADL','ADQ','KNN','NBG','NBM','SVML','SVMR','RF','ET']
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        plt.plot(x, y, 'bo')
        if (i == 0): # RL
            plt.text(x * (1 + 0.002), y * (1 + 0.001) , names[i], fontsize=12)
        elif (i == 1): # ADL
            plt.text(x * (1 - 0.018), y * (1 + 0.001) , names[i], fontsize=12)
        elif (i == 2): # ADQ
            plt.text(x * (1 + 0.003), y * (1 + 0.001) , names[i], fontsize=12)
        elif (i == 3): # KNN
            plt.text(x * (1 + 0.001), y * (1 + 0.001) , names[i], fontsize=12)
        elif (i == 5): # NBM
            plt.text(x * (1 + 0.002), y * (1 - 0.01) , names[i], fontsize=12)
        elif (i == 6): # ADL
            plt.text(x * (1 + 0.002), y * (1 + 0.001) , names[i], fontsize=12)
        elif (i == 7): # SVMR
            plt.text(x * (1 - 0.007), y * (1 - 0.013) , names[i], fontsize=12)
        elif (i == 8): # RF
            plt.text(x * (1 - 0.01), y * (1 - 0.011) , names[i], fontsize=12)
        else:
            plt.text(x * (1 + 0.001), y * (1 + 0.001) , names[i], fontsize=12)

    plt.xlim((41, 56))
    plt.ylim((38, 43))
    plt.ylabel('(a)',fontsize=20)
    plt.xticks(rotation='horizontal', fontsize=12)
    plt.yticks(rotation='horizontal', fontsize=12)
    #plt.show()

    # 10 partidas
    plt.subplot(312)
    points = [[52.87,38.23],[52.79,38.80],[46.65,42.63],[53.13,40.20],
              [48.74,41.46],[52.56,42.48],[53.06,38.60],[53.00,39.66],
              [50.65,40.33],[50.98,39.56]]
    names = ['RL','ADL','ADQ','KNN','NBG','NBM','SVML','SVMR','RF','ET']
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        plt.plot(x, y, 'rs')
        if (i == 0): # RL
            plt.text(x * (1 - 0.01), y * (1 - 0.005) , names[i], fontsize=12)
        elif (i == 1): # ADL
            plt.text(x * (1 - 0.018), y * (1 + 0.001) , names[i], fontsize=12)
        elif (i == 2): # ADQ
            plt.text(x * (1 + 0.003), y * (1 - 0.005) , names[i], fontsize=12)
        elif (i == 3): # KNN
            plt.text(x * (1 + 0.002), y * (1 + 0.001) , names[i], fontsize=12)
        elif (i == 4): # NBG
            plt.text(x * (1 + 0.003), y * (1 + 0.001) , names[i], fontsize=12)
        elif (i == 5): # NBM
            plt.text(x * (1 + 0.002), y * (1 - 0.01) , names[i], fontsize=12)
        elif (i == 6): # ADL
            plt.text(x * (1 + 0.002), y * (1 + 0.001) , names[i], fontsize=12)
        elif (i == 7): # SVMR
            plt.text(x * (1 + 0.005), y * (1 - 0.007) , names[i], fontsize=12)
        elif (i == 8): # RF
            plt.text(x * (1 - 0.012), y * (1 - 0.007) , names[i], fontsize=12)
        elif (i == 9): # ET
            plt.text(x * (1 + 0.003), y * (1 + 0.001) , names[i], fontsize=12)


    plt.xlim((41, 56))
    plt.ylim((38, 43))
    plt.ylabel('F1-score\n(b)',fontsize=20)
    plt.xticks(rotation='horizontal', fontsize=12)
    plt.yticks(rotation='horizontal', fontsize=12)

    # 19 partidas
    plt.subplot(313)
    points = [[54.73,39.34],[54.82,39.68],[41.92,41.35],[54.34,40.09],
              [50.17,41.71],[52.87,42.59],[54.35,39.12],[54.36,40.57],
              [52.14,40.56],[52.07,40.07]]
    names = ['RL','ADL','ADQ','KNN','NBG','NBM','SVML','SVMR','RF','ET']
    for i in range(len(points)):
        x = points[i][0]
        y = points[i][1]
        plt.plot(x, y, 'gD')
        if (i == 0): # RL
            plt.text(x * (1 + 0.002), y * (1 - 0.002) , names[i], fontsize=12)
        elif (i == 1): # ADL
            plt.text(x * (1 + 0.002), y * (1 - 0.002) , names[i], fontsize=12)
        elif (i == 2): # ADQ
            plt.text(x * (1 + 0.003), y * (1 - 0.005) , names[i], fontsize=12)
        elif (i == 3): # KNN
            plt.text(x * (1 + 0.002), y * (1 - 0.002) , names[i], fontsize=12)
        elif (i == 4): # NBG
            plt.text(x * (1 + 0.003), y * (1 + 0.001) , names[i], fontsize=12)
        elif (i == 5): # NBM
            plt.text(x * (1 + 0.002), y * (1 - 0.01) , names[i], fontsize=12)
        elif (i == 6): # SVML
            plt.text(x * (1 + 0.002), y * (1 - 0.008) , names[i], fontsize=12)
        elif (i == 7): # SVMR
            plt.text(x * (1 + 0.002), y * (1 - 0.002) , names[i], fontsize=12)
        elif (i == 8): # RF
            plt.text(x * (1 - 0.012), y * (1 - 0.002) , names[i], fontsize=12)
        elif (i == 9): # ET
            plt.text(x * (1 + 0.003), y * (1 - 0.005) , names[i], fontsize=12)

    plt.xlim((41, 56))
    plt.ylim((38, 43))
    plt.xlabel('Acurácia',fontsize=20)
    plt.ylabel('(c)',fontsize=20)
    plt.xticks(rotation='horizontal', fontsize=12)
    plt.yticks(rotation='horizontal', fontsize=12)
    plt.show()

def generate_significance_dataframe():

    # classificadores
    names = []
    names.append(('RL'))
    names.append(('ADL'))
    names.append(('ADQ'))
    names.append(('KNN'))
    names.append(('NBG'))
    names.append(('NBM'))
    names.append(('SVML'))
    names.append(('SVMR'))
    names.append(('RF'))
    names.append(('ET'))
    names.append(('ENS'))

    # csv com as previsões feitas pelos classificadores (instâncias)
    # no conjunto de dados de teste
    pred = pd.read_csv("../SoccerPrediction/Results/\
pred.csv", sep=';')
    df_pred = pd.DataFrame(pred, columns = ['RL','ADL','ADQ',
                                                 'ET','KNN','NBG','NBM','RF',
                                                 'SVML','SVMR','ENS'])
    df_significance = pd.DataFrame(columns=['Class1','Class2','p'])

    # loop que itera entre todos os classificadores e gera um dataframe
    # da significancia da previsão de todos entre todos
    for name in names:

        class1 = name
        dist1 = df_pred['%s' % class1].tolist()

        for name2 in names:

            class2 = name2
            dist2 = df_pred['%s' % class2].tolist()
            u, prob=stats.mannwhitneyu(dist1,dist2,alternative='two-sided')
            df_temp = pd.DataFrame({'Class1': [class1],'Class2':[class2],
                                    'p':[prob]})
            df_significance = df_significance.append(df_temp)

    df_significance.to_csv('significance.csv', sep=';')
    # numeros foram arredondados e retirou-se a notação científica pelo excel
    df_significance = pd.read_csv("../SoccerPrediction/Results/\
significance.csv", sep=';')
    df_significance = pd.DataFrame(df_significance, columns=['Class1','Class2','p'])
    significance = df_significance.pivot('Class1','Class2','p')
    print(significance)

    f, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(significance, annot=True, linewidths=.5, ax=ax)
    plt.ylabel('Classificador 1', fontsize=16)
    plt.xlabel('Classificador 2', fontsize=16)
    plt.show()

def main():

    if part_ign == 3:
	    # leitura dos CSVs
        df_england = pd.read_csv("../SoccerPrediction/Results/\
df_england_3-15_3out.csv", sep=';')
        df_england_test = pd.read_csv("../SoccerPrediction/Results/\
df_england_16-17_3out.csv", sep=';')

    if part_ign == 10:
	    # leitura dos CSVs
        df_england = pd.read_csv("../SoccerPrediction/Results/\
df_england_3-15_10out.csv", sep=';')
        df_england_test = pd.read_csv("../SoccerPrediction/Results/\
df_england_16-17_10out.csv", sep=';')

    if part_ign == 19:
	    # leitura dos CSVs
        df_england = pd.read_csv("../SoccerPrediction/Results/\
df_england_3-15_19out.csv", sep=';')
        df_england_test = pd.read_csv("../SoccerPrediction/Results/\
df_england_16-17_19out.csv", sep=';')

    # criação dos dataframes
    df_eng = pd.DataFrame(df_england, columns = ['OHE','OAE','HW',
                                                 'AW','D',
                                                 'HHGR',
                                                 'AAGR','HSHGR',
                                                 'ASAGR','DHHR',
                                                 'DAAR','VHHR','VAAR',
                                                 'HF','AF','HE','AE',
                                                 'Used','FTR'])
    df_eng_test = pd.DataFrame(df_england_test, columns = ['OHE','OAE','HW',
                                                           'AW','D',
                                                           'HHGR',
                                                           'AAGR','HSHGR',
                                                           'ASAGR','DHHR',
                                                           'DAAR','VHHR','VAAR',
                                                           'HF','AF','HE','AE',
                                                           'Used','FTR'])

    # trocando o nome das colunas
    df_eng.columns = ['EM','EV','PVM','PVV','PE','MGM',
                      'MGV','MGSM','MGSV','MEM',
                      'MEV','MVM','MVV','FM','FV','EGM','EGV',
                      'Used','R']

    df_eng_test.columns = ['EM','EV','PVM','PVV','PE','MGM',
                           'MGV','MGSM','MGSV','MEM',
                           'MEV','MVM','MVV','FM','FV','EGM','EGV',
                           'Used','R']


    # concatenando as temporadas 00/01 - 15/16 com a temporada 16/17
    dfs = [df_eng,df_eng_test]
    df = pd.concat(dfs)

    # separando em treinamento e teste
    # temporada 2002/2003 - 2014/2015
    train = df_eng.loc[df_eng.Used == 1]
    # temporada 2015/2016 - 2016/2017
    test = df_eng_test.loc[df_eng_test.Used == 1]

    # correlacao
    correlation_features(df)

    # dados de treinamento
    features = train.columns[0:17]
    # y
    y_train = (train['R'])
    # x
    x_train = train[features]

    # Acurácia em função da quantiade de features
    for k in range(1,18):
        print('\nFeatures utilizadas no treinamento:')
        features = df.columns[0:k]
        print(features)
        x_train = train[features]
        k_fold_cross_validation(x_train, y_train, 'accuracy', 10, 1, k)

    print('\nQuantidade de partidas utilizadas:', len(train))

    # comparacao de diferentes classificadores
    k_fold_cross_validation(x_train, y_train, 10, 4)

    # seleção de features (todo o dataset)
    variable_importance_RF(10000,x_train,y_train,'entropy')

    # temporada 2015/2016 - 2016/2017
    # y
    y_true = test['R']
    # x
    x_test = test[features]

    # plotando matriz de confusão de todos os classificadores
    test_and_plot_CM(x_train,y_train,x_test,y_true,part_ign)

    # scatter plot
    scatter_plot()

    # dataframe de significancia entre os resultados dos classificadores no
    # conjunto de dados de teste
    generate_significance_dataframe()


if __name__ == '__main__':
    main()
