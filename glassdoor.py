import spacy
import numpy
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MultiLabelBinarizer,LabelEncoder
import time
from sklearn.metrics import accuracy_score
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression 
from  sklearn.svm import LinearSVC
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn import model_selection
CONS_ONLY = ['haras_discrim_sexism']
class Glassdoor():
    def __init__(self):
        self.nlp = spacy.load("en_core_web_md",disable=["ner","parser"])
        self.logger = open("logs","w")


    def lemmatization(self):
        from nltk.corpus import wordnet
        synonyms = []
        antonyms = []
        for syn in wordnet.synsets(word):
                for l in syn.lemmas():
                    synonyms.append(l.name())
        return set(synonyms)
       

    def noun_normalization(self,noun): 
        from nltk.stem import PorterStemmer
        
        ps = PorterStemmer()
        return ps.stem(noun)

    
    def feauture_creation(self,phrase):
        phrase = phrase.lower().replace("-"," ").replace("!"," ").strip()
        tokens = self.nlp(phrase)
        feature_list =  [self.noun_normalization(token.text) for token in tokens if token.pos_ in ["NOUN","ADJ","VERB","ADV"]]
        return u"<>".join(feature_list)


    def data_read(self):
        pros_features = []
        pros_Y = []
        cons_features = []
        cons_Y = []
        raw = pickle.load(open("gdr_assignment_labelled.pkl","rb"))
        df = pd.DataFrame(raw)
        for index,row in df.iterrows():
            #import pdb;pdb.set_trace()
            tag, phrase = row['label'] , row['pp_sent']
            for xx in tag:
                feature_list = self.feauture_creation(phrase)
                
                cons_features.append(feature_list)
                cons_Y.append(xx)
                if xx not in CONS_ONLY:
                    pros_features.append(feature_list)
                    pros_Y.append(xx)
        return (pros_features,pros_Y,cons_features,cons_Y)


    def full_phrase_tokenizer(self,text):
        return text.split('<>')

    def rough_model_estimation(self,X,Y):
        enc = CountVectorizer(analyzer = 'word',lowercase=True,max_df=0.9,tokenizer = self.full_phrase_tokenizer,ngram_range=(1,1))
        #import pdb;pdb.set_trace()
        X = enc.fit_transform(X)
        enc_y = LabelEncoder()
        Y = enc_y.fit_transform(Y)
        models = []
        models.append(("LDA",LinearDiscriminantAnalysis()))
        models.append(("LR",LogisticRegression(multi_class="ovr",class_weight='balanced')))
        models.append(("Boosting",AdaBoostClassifier()))
        models.append(("LSVM",LinearSVC(multi_class='ovr',class_weight='balanced')))
        models.append(("perceptron",Perceptron(class_weight='balanced', random_state=0)))
        models.append(("RandomForestClassifier",RandomForestClassifier(class_weight='balanced',random_state=0)))
        for name,model in models:
            kfold = model_selection.KFold(n_splits=10,random_state=7)
            for score in ["accuracy"]:
                print("checking %s:%s"%(name,score))
                start = time.clock()
                try:
                    cv_results = model_selection.cross_val_score(model,X,Y,cv=kfold,scoring=score)
                except TypeError:
                    x_dense = X.todense()
                    cv_results = model_selection.cross_val_score(model,x_dense,Y,cv=kfold,scoring=score)
                print("time taken:")
                print(time.clock() - start)
                msg = "%s: %f (%f) :%s" % (name, cv_results.mean(), cv_results.std(),score)
                print(msg)

    def hyper_parameter_tuning(self,X_train,Y_train):
        print("grid search cell started")
        from sklearn.model_selection import GridSearchCV
        tuned_parameters = [{"C":[1000,100,10,1.0,0.1],"penalty":['l1','l2']}]
        clf = GridSearchCV(LogisticRegression(multi_class="ovr",class_weight='balanced'), tuned_parameters, cv=5,scoring='accuracy')
        clf.fit(X_train, Y_train)
        print("Best parameters set found on development set:")    
        print(clf.best_params_)
        print("Grid scores on development set:")
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))
        return clf.best_params_

    def final_model_fitting(self,X_train,Y_train,best_params):
        from sklearn.metrics import classification_report
        from sklearn.metrics import confusion_matrix
        from sklearn.metrics import accuracy_score
        from sklearn.ensemble import BaggingClassifier
        #import pdb;pdb.set_trace()
        C_p = best_params['C']
        penalty_p = best_params['penalty']
        clf = LogisticRegression(C=C_p,penalty = penalty_p,multi_class='ovr',class_weight='balanced')
        #clf = BaggingClassifier(base_estimator=clfL,n_estimators=10)
        clf.fit(X_train, Y_train)
        return clf

    def model_reports(self,clf,X_test,Y_test):
        predictions = clf.predict(X_test)
        print(accuracy_score(Y_test, predictions))
        print(confusion_matrix(Y_test, predictions))
        print(classification_report(Y_test, predictions))
    
    def debugging_logistic_hypothesus(self):
        feature_names = enc.get_feature_names()
        feature_vectors = clf.coef_
        for index,feature_vector in enumerate(feature_vectors):
            features = [(feature_names[feature_index],feature_weight) for feature_index,feature_weight in enumerate(feature_vector)]
            feature_sorted_by_weight = sorted(features, key=lambda tup: tup[1],reverse=True)
            id_type = enc_y.inverse_transform([index])
            self.logger.write("%s\n"%id_type[0])
            self.logger.write("%s\n"%u"<>".join(["%s|%s"%(x,y) for x,y in feature_sorted_by_weight[:20]]))
 
    def review_breaking_logic(self,review,clf,enc,enc_y):
        ans = []
        reviews = review.split("\n")
        if len(reviews) < 2:
            reviews = review.split(".")
        if len(reviews) < 2:
            reviews = review.split(",")
        for row in reviews:
            features = self.feauture_creation(row)
            #f_list.append(features)
            features = [features]
            final_x = enc.transform(features) 
            predictions = clf.predict(final_x)
            ans.append("%s<>%s"%(row,enc_y.inverse_transform(predictions)))
        return ans
            #print(row)
            #print(enc_y.inverse_transform(predictions))

    def phrase_category_prediction(self,pros_clf,cons_clf,enc_pros,encp_y,enc_cons,encc_y):
        raw = pickle.load(open("gdr_assignment_pros_cons.pkl","rb"))
        df = pd.DataFrame(raw)
        #import pdb;pdb.set_trace()
        # pros = df[["pros"]]
        self.output_file = open("glassdoor_output","w")
        for i,row in df.iterrows():
            pros =  row["pros"].strip(".\n")
            cons =  row["cons"].strip(".\n")
            pros_category_list = self.review_breaking_logic(pros,pros_clf,enc_pros,encp_y)
            cons_category_list = self.review_breaking_logic(cons,cons_clf,enc_cons,encc_y)
            self.output_file.write("Pros:\n%s\n"%("\n".join(pros_category_list)))
            self.output_file.write("Cons:\n%s\n"%("\n".join(cons_category_list)))
            self.output_file.write("---------------------------------------------\n") 
            


    def training(self,features,Y):
        validation_size = 0.2
        seed =7
        X_train, X_test, Y_train, Y_test = model_selection.train_test_split(features, Y, test_size=validation_size, random_state=seed)
        #self.rough_model_estimation(features,Y)
        enc = CountVectorizer(analyzer = 'word',lowercase=True,max_df=0.9,tokenizer = self.full_phrase_tokenizer,ngram_range=(1,1))
        X_train = enc.fit_transform(X_train)
        enc_y = LabelEncoder()
        Y_train = enc_y.fit_transform(Y_train)
        best_params = self.hyper_parameter_tuning(X_train,Y_train)
        clf = self.final_model_fitting(X_train,Y_train,best_params)
        X_test = enc.transform(X_test)
        Y_test = enc_y.transform(Y_test)
        self.model_reports(clf,X_test,Y_test)
        #f_2d=numpy.array([numpy.array(xi) for xi in features])
        #import pdb;pdb.set_trace() 
        return (clf,enc,enc_y)
        


if(__name__ == "__main__"):
    g = Glassdoor()
    pros_x, pros_y, cons_x, cons_y = g.data_read()
    pros_classifier,enc_pros,encp_y = g.training(pros_x,pros_y)
    cons_classifier,enc_cons,encc_y = g.training(cons_x,cons_y)
    g.phrase_category_prediction(pros_classifier,cons_classifier,enc_pros,encp_y,enc_cons,encc_y) 
        
                
         
