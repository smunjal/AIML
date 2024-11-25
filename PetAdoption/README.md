# Module Capstone Project
By Sumant Munjal

Link to Notebooks: 
- Main Notebook :  https://github.com/smunjal/AIML/blob/main/PA_11/module_11.ipynb
- Function Notebook :  https://github.com/smunjal/AIML/blob/main/PA_11/CommonFunction.ipynb
    - Note: Functions notebook is loaded from Main

# Problem

The goal of the project is to develop a model to predict the adoptability of pets - specifically, how quickly is a pet adopted? If successful, so that it can be adapted into AI tools that will guide shelters and rescuers around the world on improving their pet profiles' appeal, reducing animal suffering and euthanization.

### Findings - SO Far

- The problem is a multiclass classification problem
- Random Forest is the best classifier with test score as 67%, Followed by KNN - 48% and DecisionTree - 46%
- For one of multiclass the micro score for f1 metric is 19%, but precision is 94%. That is the reason for overall f1-micro score to be around 63%
- Need to dive in what is going on here


### Next Steps
1. This is still a WIP(Work In Progress)
2. We have baseline scores for basic models board, we want to try to perform some tuning using hyper-parameters
3. Sentimental analysis of description test needs to be performed
4. Possibly do image analysis also
4. Need to do perform predictions for the test dataset 


