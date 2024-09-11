# Module 11
By Sumant Munjal

Link to Notebooks: 
- Main Notebook :  https://github.com/smunjal/AIML/blob/main/PA_11/module_11.ipynb
- Function Notebook :  https://github.com/smunjal/AIML/blob/main/PA_11/CommonFunction.ipynb
    - Note: Functions notebook is loaded from Main

# Problem

From a data science perspective, the task is to identify the primary factors influencing used car prices at the dealership.


### Findings
- Sales have been growing at a steady pace since 2000.
- 2017/2018 were are best year for sales where we sold in excess of 35K cars in our dealership, since then car sales have declined.
- During best years the dealership was selling lot of newer cars/low mileage and high price.
- Buyers are also looking for alternative fuel types other than gas - hybrid/electric
- Cars buyers also look at the manufacturer of cars
    - Japanese makers Toyota/Honda/Nissan are big sellers, since they provide multiple fuel type options of electric/hybrid combinations
    - Among american makers - Ford/Chevy are most wanted   

#### So key takeaway, to make more money let's sell cars which are expensive and these factors makes cars more expensive(in order)

1. Newer model cars
2. Alternate fuel type
    - Electric/Hybrid 
3. Low Mileage
4. Right manufacturers
    - The big three Japs(Toyota/Honda/Nissan), lets stick with 
        -  Sedan's
    - and American(Ford/Chevy/GMC)
        - Trucks/Pickup's
5. Transmission     
    - Newer/Other types of transmission types are high seller 
6. Clean title     

### Next Steps
1. WARNING:  It takes a long time to run the notebook on my MAC, some GridSearch Steps takes over 15-20 mins to feature select, so make it faster
2. The model scores are not that great in range of ~20%, so lot of scope of improvement
3. Chose different/better encoders OneHot/Label rather than chosen CountEncoder for this test 

