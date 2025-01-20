# Module Capstone Project
By Sumant Munjal

Link to Notebooks: 
- Main Notebook :  https://github.com/smunjal/AIML/blob/main/PetAdoption/capstone-petfinder-EDA.ipynb
- Main Notebook :  https://github.com/smunjal/AIML/blob/main/PetAdoption/capstone-petfinder.ipynb
- Common Functions :  https://github.com/smunjal/AIML/blob/main/PetAdoption/common_fn.py
    - Note: Functions are loaded from Main Notebook

# Problem

The goal of the project is to develop a model to predict the adoptability of pets - specifically, how quickly is a pet adopted? If successful, so that it can be adapted into AI tools that will guide shelters and rescuers around the world on improving their pet profiles' appeal, reducing animal suffering and euthanization.

### Findings 

- The problem is a multiclass classification problem
- AdoptionSpeed(target) has 5 classes,
    - 0 : '1st Day'
    - 1 : '1st Week'
    - 2 : '1st Month'
    - 3 : '2nd & 3rd month'
    - 4 : 'Not Adopted'
- The classes are also imbalanced, so f1-score with average as 'micro', to focus on individual classes
- Various tuning approaches - SMOTE to balance classes, Hyperparameter tuning, Threshold tuning were done. The f1-score improvement was very small, but class-wise predictions were improved and hence that was the criterion used for eval  
- Random Forest is the found the classifier with test score as 40%, Followed by XGBoost - 39% and GradientBoosting - 39%
- Also tried a StackingClassifier - but did not help in overall f1-score
- Observations on the final predictions done using full training dataset shows
    - We see a big jump in f1-score from 40% we have been seeing across all tests to 60%.
    - It might be attributed that we are using the entire training dataset instead of train-test split we did earlier
    - Also we are using training for predictions, since we don't have predictions for the test dataset for validation
    - Confusion matrix shows the diagonal having the max for most of classes, apart from Class 0 as we have been observing in all tests
- Looks like we made a good choice of model selection
- The final submission.csv with predictions is generated in 'data2/test', which contains PetID,AdoptionSpeed 


### Next Steps
4. Possibly do image analysis, since that dataset was also provied



