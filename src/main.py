'''
You will run this problem set from main.py, so set things up accordingly
'''

import pandas as pd
import etl
import preprocessing
import logistic_regression
import decision_tree
import calibration_plot


# Call functions / instanciate objects from the .py files
def main():

    # PART 1: Instanciate etl, saving the two datasets in `./data/`
    etl.run_etl()

    # PART 2: Call functions/instanciate objects from preprocessing
    preprocessing.run_preprocessing()

    # PART 3: Call functions/instanciate objects from logistic_regression
    df_arrests_train, df_arrests_test, lr_model = logistic_regression.run_logistic_regression(df_arrests)
    
    # PART 4: Call functions/instanciate objects from decision_tree
    df_arrests_test = decision_tree.run_decision_tree(df_arrests_train, df_arrests_test)

    # PART 5: Call functions/instanciate objects from calibration_plot
    # for logistic regression
    print("Calibration plot: logistic regession")
    calibration_plot.calibration_plot(df_arrests_test["y"], df_arrests_test["pred_lr"], n_bins=5)
    # for decision tree
    print("Calibration PlotL decision tree")
    calibration_plot.calibration_plot(df_arrests_test["y"], df_arrests_test["pred_dt"], n_bins=5)
    print("Which model is more calibrated?")
    print("Based off the plots it should be logistic regression")

    


if __name__ == "__main__":
    main()