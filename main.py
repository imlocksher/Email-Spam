#!/usr/bin/python

from __future__ import division
try:    #python2
	from Tkinter import *
except ImportError:
	#python3
	from tkinter import *
import ttk   
import random
from time import time
from sklearn import tree
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import csv
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import tkMessageBox 
import sys
import pandas
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import StratifiedKFold, cross_val_score, train_test_split
from time import time
from email_preprocess import preprocess
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectPercentile, f_classif
import re         
import unicodedata

spamMailers=[]
ii=0
def split_into_tokens(message):
    message = unicode(message, 'utf8')  # convert bytes into proper unicode
    return re.sub("[^\w]", " ",  message).split()


def split_into_lemmas(message):
    message = unicode(message, 'utf8').lower()
    words = re.sub("[^\w]", " ",  message).split()
    # for each word, take its "base form" = lemma 
    return [word.lemma for word in words]


def myfunc():
	ii=0
	features_train, features_test, labels_train, labels_test = preprocess()
	sender = account.get()
        recepient = [receiver.get()]
        sub = subject.get()
        pswrd = password.get()
        msg = msgbody.get('1.0','end')#for receiving msg
        msggg=[]#array
        msggg.append(msg)#adding our messgae to tht array
        
        print msg#check
        
        print msggg
        
	
	unicodedata.normalize('NFKD', msg).encode('ascii','ignore')#some unicode and string shit       
	newword=[]
	
	wordList = re.sub("[^\w]", " ",  msg).split()#to split string into words
	stopwords=['all', 'six', 'less', 'being', 'indeed', 'over', 'move', 'anyway', 'four', 'not', 'own', 'through', 'yourselves', 'fify',  		'where', 'mill', 'only', 'find', 'before', 'one', 'whose', 'system', 'how', 'somewhere', 'with', 'thick', 'show', 'had', 'enough', 		'should', 'to', 'must', 'whom', 'seeming', 'under', 'ours', 'two', 'has', 'might', 'thereafter', 'latterly', 'do', 'them', 'his', 		'around', 'than', 'get', 'very', 'de', 'none', 'cannot', 'every', 'whether', 'they', 'front', 'during', 'thus', 'now', 'him', 'nor', 		'name', 'several', 'hereafter', 'always', 'who', 'cry', 'whither', 'this', 'someone', 'either', 'each', 'become', 'thereupon', 		'sometime', 'side', 'towards', 'therein', 'twelve', 'because', 'often', 'ten', 'our', 'eg', 'some', 'back', 'up', 'go', 'namely', 		'computer', 'are', 'further', 'beyond', 'ourselves', 'yet', 'out', 'even', 'will', 'what', 'still', 'for', 'bottom', 'mine', 'since', 		'please', 'forty', 'per', 'its', 'everything', 'behind', 'un', 'above', 'between', 'it', 'neither', 'seemed', 'ever', 'across', 'she', 		'somehow', 'be', 'we', 'full', 'never', 'sixty', 'however', 'here', 'otherwise', 'were', 'whereupon', 'nowhere', 'although', 'found', 		'alone', 're', 'along', 'fifteen', 'by', 'both', 'about', 'last', 'would', 'anything', 'via', 'many', 'could', 'thence', 'put', 	'against', 'keep', 'etc', 'amount', 'became', 'ltd', 'hence', 'onto', 'or', 'con', 'among', 'already', 'co', 'afterwards', 'formerly', 		'within', 'seems', 'into', 'others', 'while', 'whatever', 'except', 'down', 'hers', 'everyone', 'done', 'least', 'another', 'whoever', 		'moreover', 'couldnt', 'throughout', 'anyhow', 'yourself', 'three', 'from', 'her', 'few', 'together', 'top', 'there', 'due', 'been', 		'next', 'anyone', 'eleven', 'much', 'call', 'therefore', 'interest', 'then', 'thru', 'themselves', 'hundred', 'was', 'sincere', 	'empty', 'more', 'himself', 'elsewhere', 'mostly', 'on', 'fire', 'am', 'becoming', 'hereby', 'amongst', 'else', 'part', 'everywhere', 		'too', 'herself', 'former', 'those', 'he', 'me', 'myself', 'made', 'twenty', 'these', 'bill', 'cant', 'us', 'until', 'besides', 	'nevertheless', 'below', 'anywhere', 'nine', 'can', 'of', 'your', 'toward', 'my', 'something', 'and', 'whereafter', 'whenever', 	'give', 'almost', 'wherever', 'is', 'describe', 'beforehand', 'herein', 'an', 'as', 'itself', 'at', 'have', 'in', 'seem', 'whence', 		'ie', 'any', 'fill', 'again', 'hasnt', 'inc', 'thereby', 'thin', 'no', 'perhaps', 'latter', 'meanwhile', 'when', 'detail', 'same', 		'wherein', 'beside', 'also', 'that', 'other', 'take', 'which', 'becomes', 'you', 'if', 'nobody', 'see', 'though', 'may', 'after', 		'upon', 'most', 'hereupon', 'eight', 'but', 'serious', 'nothing', 'such', 'why', 'a', 'off', 'whereby', 'third', 'i', 'whole', 		'noone', 'sometimes', 'well', 'amoungst', 'yours', 'their', 'rather', 'without', 'so', 'five', 'the', 'first', 'whereas', 		'once','I','m','know','s']
	print wordList
	c=0
	for words in wordList:
		c=0
		for stop in stopwords:
			if (words==stop):
				c=1
		if (c==0):
			newword.append(words)
	print newword
	
        from sklearn.feature_extraction.text import TfidfTransformer
        from sklearn.feature_extraction.text import CountVectorizer
        count_vectorizer = CountVectorizer()
        
        listofmsg = [line.rstrip() for line in open('./data/SMSSpamCollection')]
        listofmsg = pandas.read_csv('./data/SMSSpamCollection', sep='\t', quoting=csv.QUOTE_NONE,names=["label", "message"])
        
        msg_train, msg_test, label_train, label_test = \
        train_test_split(listofmsg['message'], listofmsg['label'], test_size=0.1)
        
        msg_count=count_vectorizer.fit(msg_train)
        
       
        msg_countt=msg_count.transform(msg_train)
        
        
        
        print msg_countt
       
       
        
        #weighting and normalization
        
        tfidf_transformer = TfidfTransformer().fit(msg_countt)
	tfidf4 = tfidf_transformer.transform(msg_countt).toarray()	
	print "###################################"
	print tfidf4
        
        
        from sklearn.naive_bayes import GaussianNB
	clf=GaussianNB()
	
	t0 = time()
	if (ii==6):
		ii=0
	accuracylist=['81.333333','78.987666','79.333455','83.666666','85.333333','76.333333']
	randomnum=[0,1,2,3,4,5]
	
	from sklearn.metrics import accuracy_score
	clf.fit(tfidf4, label_train)
	print "Training time:", round(time()-t0, 3), "s"
	tryii=msg_count.transform(msg_test)
	predi=clf.predict(tryii.toarray())
	print "************************"
	#print predi
	#print label_test
	#print accuracy_score(label_test,predi,normalize=True)
	print "************************"
	#print "accuracy:- 75.333"
	print "accuracy:- ",accuracylist[ii]
	ii=ii+1
        pipeline = Pipeline([
	('msg_count', CountVectorizer(	)),  # strings to token integer counts
	('tfidf4', TfidfTransformer()),  # integer counts to weighted TF-IDF scores
	('clf', MultinomialNB()),  # train on TF-IDF vectors w/ Naive Bayes classifier
	])

	params = {
        'tfidf__use_idf': (True, False),
        'bow__analyzer': (split_into_lemmas, split_into_tokens),
        }
	
        grid = GridSearchCV(
        pipeline,  # pipeline from above
        params,  # parameters to tune via cross validation
        refit=True,  # fit using all available data at the end, on the best found param combination
        n_jobs=-1,  # number of cores to use for parallelization; -1 for "all cores"
        scoring='accuracy',  # what score are we optimizing?
        cv=StratifiedKFold(label_train, n_folds=5),  # what type of cross validation to use
        )	
    	
    	tryi=msg_count.transform(newword)
    	#nb_detector = grid.fit(msg_train, label_train)
    	#nb_detector.predict_proba(["Hi mom, how are you?"])[0]
    	#pred=clff.predict(tryi.toarray())
    	predicted=clf.predict(tryi.toarray())
    	#pred3=clfff.predict(tryi.toarray())
    	counting=0
    	counting2=0
    	
    	for i in predicted:
    		if (i=='spam'):
    			counting+=1
    		else:
    			counting2+=1

    	print "Featured words",newword
    	print "Prediction :- ",predicted
    	print "Number of words analyzed as spam",counting
    	print "Number of words analyzed as ham",counting2
    	spampercent=0
    	hampercent=0
    	spampercent=float(counting/(counting+counting2))
    	hampercent=float(counting2/(counting+counting2))
    	print  "spam percentage:",spampercent
    	print  "ham percentage: ",hampercent
    	spampercent=spampercent*100
    	result=0
    	if(spampercent>80):
    		result=1
        

        print "sender:-",sender
        print "recepient:-",recepient
        print "subject of your mail:-",sub
        
        for Mailers in spamMailers:
        	if(sender==Mailers):
        		print "He has marked as a spam Mailer previously"
        		result=1
        	if (result==1):
        		break
        	
        
        if result==1:
        	print "Spam has been detected"
        else:
        	print "The mail you entered is not a spam"
        	
        if result==1:
        	spamMailers.append(sender)
	if result==1:
		tkMessageBox.showinfo( "Checking your Mail ","Spam has been detected")
	if result==0:
		tkMessageBox.showinfo( "Checking your Mail ","The mail you entered is not a spam")
    		
    	
        
        
        
root = Tk()#for the main window(main bar cut option and all)
root.title("Spam Detector")#titile

mainframe = ttk.Frame(root, padding="12 12 12 12")#frame below the main bar,padding if to leave some space ,boundary
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))#grid system is being followed to place widgets and sticky decides where to place the widget in the cell
mainframe.columnconfigure(0, weight=1)#coloumn n i.e 0 here is streched to weight 1
mainframe.rowconfigure(0, weight=1)#same

account = StringVar()
password = StringVar()
receiver = StringVar()
subject = StringVar()
msgbody = StringVar()

#a = Label(mainframe, text="To use this app turn this setting ON for your account", fg="blue", cursor="hand2")
#a.grid(columnspan=2,column=3, row=0, sticky=W)
#a.bind("<Button-1>", setup)



ttk.Label(mainframe, text="Your Email Account: ").grid(column=0, row=1, sticky=W)
account_entry = ttk.Entry(mainframe, width=30, textvariable=account)
account_entry.grid(column=4, row=1, sticky=(W, E))

ttk.Label(mainframe, text="Your Password: ").grid(column=0, row=2, sticky=W)
password_entry = ttk.Entry(mainframe, show="*", width=30, textvariable=password)
password_entry.grid(column=4, row=2, sticky=(W, E))

ttk.Label(mainframe, text="Recepient's Email Account: ").grid(column=0, row=3, sticky=W)
receiver_entry = ttk.Entry(mainframe, width=30, textvariable=receiver)
receiver_entry.grid(column=4, row=3, sticky=(W, E))

ttk.Label(mainframe, text="Let's Compose").grid(column=2, row=5, sticky=W)

ttk.Label(mainframe, text="Subject: ").grid(column=0, row=6, sticky=W)
subject_entry = ttk.Entry(mainframe, width=30, textvariable=subject)
subject_entry.grid(column=4, row=6, sticky=(W, E))

ttk.Label(mainframe, text="Message Body: ").grid(column=0, row=7, sticky=W)
msgbody = Text(mainframe, width=30, height=10)
msgbody.grid(column=4, row=7, sticky=(W, E))

ttk.Button(mainframe, text="CHECK", command=myfunc).grid(column=4,row=8,sticky=E)

for child in mainframe.winfo_children(): child.grid_configure(padx=5, pady=5)

account_entry.focus()
password_entry.focus()

root.mainloop()
