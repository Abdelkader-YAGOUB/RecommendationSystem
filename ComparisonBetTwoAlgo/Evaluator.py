from EvaluationData import EvaluationData
from EvaluatedAlgorithm import EvaluatedAlgorithm
from beautifultable import BeautifulTable

# Color for text OUtPUT
BeginRED = '\033[91m'
EndRED = '\033[0m'

BeginBgRED = '\033[41m'
EndBgRED = '\033[0m'

BeginGREEN = '\033[92m'
EndGREEN = '\033[0m'

BeginBgGREEN ='\033[42m'
EndBgGREEN ='\033[0m'

BeginYELLO = '\033[93m'
EndYELLO = '\033[0m'

BeginBgYELLO = '\033[43m'
EndBgYELLO = '\033[0m'

BeginBLUE = '\033[94m'
EndBLUE = '\033[0m'

BeginBgBLUE = '\033[44m'
EndBgBLUE = '\033[0m'


class Evaluator:
    
    algorithms = []
    
    def __init__(self, dataset, rankings):
        ed = EvaluationData(dataset, rankings)
        self.dataset = ed
        
    def AddAlgorithm(self, algorithm, name):
        alg = EvaluatedAlgorithm(algorithm, name)
        self.algorithms.append(alg)
        
    def Evaluate(self, doTopN):
        results = {}
        for algorithm in self.algorithms:
            print(BeginGREEN + "Evaluating " + EndGREEN, algorithm.GetName(), BeginGREEN + "..." + EndGREEN)
            results[algorithm.GetName()] = algorithm.Evaluate(self.dataset, doTopN)

        # Print results table
        table = BeautifulTable(maxwidth=100)
        if (doTopN):
            table.columns.header = ["Algorithm", "RMSE", "MAE", "HR", "cHR", "ARHR", "Coverage" , "Diversity", "Novelty"]
            for (name, metrics) in results.items():
                table.rows.append([name, metrics["RMSE"], metrics["MAE"], metrics["HR"], metrics["cHR"], metrics["ARHR"], metrics["Coverage"], metrics["Diversity"], metrics["Novelty"]])
        else:
            table.columns.header = ["Algorithm", "RMSE", "MAE"]
            for (name, metrics) in results.items():
                table.rows.append([name, metrics["RMSE"], metrics["MAE"]])
        table.set_style(BeautifulTable.STYLE_BOX_ROUNDED)
        print(table)
        
    def SampleTopNRecs(self, ml, testSubject=85, k=10):
        
        for algo in self.algorithms:
            print(BeginGREEN + "\nUsing recommender " + EndGREEN, algo.GetName())
            
            print(BeginGREEN + "\nBuilding recommendation model..." + EndGREEN)
            trainSet = self.dataset.GetFullTrainSet()
            algo.GetAlgorithm().fit(trainSet)
            
            print(BeginGREEN + "Computing recommendations..." + EndGREEN)
            testSet = self.dataset.GetAntiTestSetForUser(testSubject)
        
            predictions = algo.GetAlgorithm().test(testSet)
            
            recommendations = []
            
            print (BeginYELLO + "\nWe recommend:" + EndYELLO)
            for userID, movieID, actualRating, estimatedRating, _ in predictions:
                intMovieID = int(movieID)
                recommendations.append((intMovieID, estimatedRating))
            
            recommendations.sort(key=lambda x: x[1], reverse=True)
            
            for ratings in recommendations[:10]:
                print(ml.getMovieName(ratings[0]), ratings[1])
    
    