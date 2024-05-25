This is the evaluation code, you can use this to evaluate your tracking algorithm.

Before using the code, please use the following command to install the required enviroments.
1. conda create -n mtmceval python=3.8
2. conda activate mtmceval
3. pip install cmake
4. pip install -r requirements.txt
5. To verify your installation: run "python3 eval.py gt/74.txt ../example_tracking_result.txt", you should be able to get an example evaluation results on the validation set.


Now, you should be able to run tracking evaluation on your own!

Please use this command to run evaluation: "python3 eval.py <ground_truth> <prediction>" (Remember to use the mtmceval conda enviroment!) 

The <ground_truth> and <prediction> should be set as the path of your txt file.
