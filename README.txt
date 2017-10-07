                            Hotel Recommendation System		
                Arjun Dhuliya (amd5300)	Siddharth Subramanian (ss6813)
                        Advisor : Yuxiao Huang (yhvcs@rit.edu)
--------------------------------------------------------------------------------------
The main file is called hotel_recommender.py which performs the following operations
    -   Data preparation & down-sampling
    -   Building a model
    -   Running of 4 machine learning algorithms
    -   Prediction of clusters and calculation of accuracy

The original dataset for this project can be found at :
    https://www.kaggle.com/c/expedia-hotel-recommendations/data

The down sampled version of the data used is submitted as out.csv
The script python packages pandas scikit-learn packages
The data and the .py script should be present in the same folder
The script also writes the model for each machine learning algorithm as a .dat file which can 
be reloaded into the script using pickle. 