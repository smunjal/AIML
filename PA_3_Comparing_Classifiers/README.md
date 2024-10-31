# Module 17
By Sumant Munjal

Link to Notebooks:  https://github.com/smunjal/AIML/blob/main/PA_3_Comparing_Classifiers/prompt_III.ipynb

# Problem

The problem dataset represents data from a Portugese banking institution and is a collection of the results of 17 marketing campaigns that
occurred between May 2008 and November 2010,During these phone campaigns, an attractive long-term deposit was offered, this was captured(Yes/No) in the target feature 'y'. The goal for this problem is to compare the performance prediction of the various classifiers namely K Nearest Neighbor, Logistic Regression, Decision Trees, and Support Vector Machines.

### Findings/Takeaways

1. The target features 'y' has a split class ratio of 80%(No) to 20%(Yes), which signifies it is imbalabced dataset, So
    - 'accuracy' is not the correct metric for benchmarking, since it become easy to get a high accuracy score by simply classifying all observations as the majority class, which in this case is 0(No subscription of term deposit)
    - If the goal is for reducing Type-I/False Positive's, we can choose 'precision' as the performance metric
    - If the goal is for reducing For Type-II erros,  we can choose 'recall'
    - For the classifier comparison, 'F1-score' was chosen as the performance metric which represents harmonic mean of Precision and Recall, and gives a combined idea about these two metrics.
    - However a comparison is also done for FPR/TPR across all f1-score/Precision/Recall 
2. When out-of-box or classifiers with all default values where run and compared 
    - Logistic Regression is the best classifier with test score as 50%, Followed by DecisionTree - 46% and SVM - 38%
    - Decision Tree has a train score as 100% and Test score is 46%, explain tha model is overfitted
3. After tuning/configuring classifiers using hyperparameters, the comparisons highlights 
     - SVM was the best classifier where f1-score improved from 38% to 57%, a 50% improvement
     - DecisionTree is not overfitted this time and score improved from 46-56%
     - LogisticRegression score did not change at all
     - kNN(K-nearest neighbour) is overfitted, since train score was 1
4. The FPR/TPE comparison for SVM - f1-score/precision/recal shows  
     -  For 'precision' we see that False positive Rate has gone down, but positive rate is also not high, with a high AUC=0.92
     -  SVM with 'f1' metric seems to be the best, since it has balance between FPR(lower) and TPR(higher) ad with AUC = 0.93
5. It was observed the scores varied a major difference, if we set `prediction = True`(global variable), which drops Feature 11 - 'duration'. Is some target data-leaking happening when it is included ?   
     

### Next Steps
1. We can try to rerun by synthetically balancing the target's minority/positive class? SMOTE will balance the ratios
2. Test with other encoder like 'TargetEncoder' to see the difference and performance of the classifiers?


