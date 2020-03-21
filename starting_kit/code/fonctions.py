from sklearn.ensemble import IsolationForest

def supprOutlier(X,y,retIndex=False):
   clf = IsolationForest(random_state=0).fit(X)
   res = clf.predict(X)

   indexOutlier = []
   for i in range (0,len(res)):
        if res[i]==-1:
            indexOutlier.append(i)

   pourcentOutlier = len(indexOutlier) / len(res) * 100
   print("Pourcentage d'outlier: {0:.3f}%:".format(pourcentOutlier))

   if retIndex:
        return np.delete(X,indexOutlier,axis=0),np.delete(y,indexOutlier,axis=0),indexOutelier

        #delete: Supprime les LIGNES d'indices dans indexOutlier
   return np.delete(X,indexOutlier,axis=0),np.delete(y,indexOutlier,axis=0)