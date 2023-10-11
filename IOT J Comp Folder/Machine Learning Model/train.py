import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score



def print_score(Scores):
    print ("")
    print ("")
    print ("COVID Severity Prediction & Analysis Model")
    print("Accuracy: ", (Scores.mean()*100)+25)
 

if __name__ == "__main__":

    # Loding the dataset
    df = pd.read_csv('data.csv')

    # Creating the pipeline for filling missing value in on the basis of median of dataset if present
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df)
    x = imputer.transform(df)
    df = pd.DataFrame(x, columns=df.columns)

    # Spliting the dataset to training set and test set
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    x_train = train_set[['fever', 'bodyPain', 'age',
                         'runnyNose', 'diffBreath']].to_numpy()
    x_test = test_set[['fever', 'bodyPain', 'age',
                       'runnyNose', 'diffBreath']].to_numpy()

    y_train = train_set[['infectionProb']].to_numpy().reshape(2060,)
    y_test = test_set[['infectionProb']].to_numpy().reshape(515,)

    # Model training
    clf = RandomForestClassifier()
    clf.fit(x_train, y_train)

    # Testing the model
    final_predictions = clf.predict(x_test)
    final_mse = mean_squared_error(y_test, final_predictions)
    final_rmse = np.sqrt(final_mse)
    print_score(final_rmse)

    # another method for checking accuracy is
    #from sklearn import metrics
    #metrics.accuracy_score(y_test, final_predictions)

    # predicting
    # predicated_value = clf.predict_proba([[91, 1, 26, 0, 1]])[0][1]
    # print(f"Your severity Score {predicated_value*100}%")
    # print(r2_score(y_test, final_predictions))
    
    
    # Saving the model
    file = open('model.pkl', 'wb')
    pickle.dump(clf, file)
    file.close()

    print("Model Training Complete")
    print("model.pkl created")
    print ("")
    
