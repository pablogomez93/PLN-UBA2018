from tass import InterTASSReader
from tass import GeneralTASSReader
from collections import defaultdict


def printStatistics(reader, corpusName):
    X = list(reader.X())  # iterador sobre los contenidos de los tweets
    y = list(reader.y())  # iterador sobre las polaridades de los tweets

    polarities = defaultdict(int)
    for polarity in y:
        polarities[polarity] += 1

    print()
    print(corpusName)
    print('Tweets totales:', len(X))
    print('Tweets P:', polarities['P'])
    print('Tweets N:', polarities['N'])
    print('Tweets NEU:', polarities['NEU'])
    print('Tweets NONE:', polarities['NONE'])
    print()


printStatistics(InterTASSReader('TASS/InterTASS/tw_faces4tassTrain1000rc.xml'), "InterTASS")
printStatistics(GeneralTASSReader('TASS/GeneralTASS/general-tweets-train-tagged.xml'), "GeneralTASS")
