# Load libraries
import pandas
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
      
def print_menu():    
    print 30 * "-" , "MENU" , 30 * "-"
    print "1. Load dataset"
    print "2. Show Box & Whisper Plots"
    print "3. Show Histograms"
    print "4. Split-out validation dataset"
    print "5. Spot check algorithms" 
    print "6. Exit"
    print 67 * "-"
  
loop=True      
  
while loop:          ## While loop which will keep going until loop = False
    print_menu()    ## Displays menu
    choice = input("Enter your choice [1-6]: ")
     
    if choice==1:     
        print "Dataset loaded (iris.data)"
        # Load dataset
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
        dataset = pandas.read_csv(url, names=names)
    elif choice==2:
        print "Showing Box & Whisper Plots"
        # box and whisker plots
        dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
        plt.show()
    elif choice==3:
        print "Showing Histograms"
        # histograms
        dataset.hist()
        plt.show()
    elif choice==4:
        print "Split-out validation dataset"
        # Split-out validation dataset
        array = dataset.values
        X = array[:,0:4]
        Y = array[:,4]
        validation_size = 0.20
        seed = 7
        X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)            
	# Test options and evaluation metric
        seed = 7
        scoring = 'accuracy'
    elif choice==5:
        print "Spot check algorithms"
        # Spot Check Algorithms
        models = []
        models.append(('LR', LogisticRegression()))
        models.append(('LDA', LinearDiscriminantAnalysis()))
        models.append(('KNN', KNeighborsClassifier()))
        models.append(('CART', DecisionTreeClassifier()))
        models.append(('NB', GaussianNB()))
        models.append(('SVM', SVC()))
        # evaluate each model in turn
        results = []
        names = []
        for name, model in models:
                kfold = model_selection.KFold(n_splits=10, random_state=seed)
                cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
                results.append(cv_results)
                names.append(name)
                msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                print(msg)
    elif choice==6:
        print "Exiting program"
        import sys
        sys.exit()
    else:
        # Any integer inputs other than values 1-6 we print an error message
        raw_input("Wrong option selection. Enter any key to try again..")
