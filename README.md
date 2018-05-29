# End-to-End-Memory-Networks

The following results have been provided for Linear Start, Positional Encoding, Random Noise, 3 Hops and Joint training since this was the best reported model of the paper. 

The dataset used is babI https://research.fb.com/downloads/babi/ (1k english training samples)

## Loss curves
![image](https://user-images.githubusercontent.com/4141117/40680834-3e441660-633c-11e8-91ff-ccc35317c75c.png)

The following are the results on the test set.
## Test Results

Task  |  Training Acc.  
------|-----------------
1     |  0.995           
2     |  0.816           
3     |  0.729           
4     |  0.759          
5     |  0.83          
6     |  0.59           
7     |  0.834           
8     |  0.873           
9     |  0.651          
10    |  0.98           
11    |  0.98           
12    |  1.00           
13    |  0.97           
14    |  0.67           
15    |  0.23           
16    |  0.5           
17    |  0.54          
18    |  0.53           
19    |  0.11           
20    |  0.97           
mean  |  0.706           

## Instructions to run the code:
1. Download the babI dataset from the link provided above.
2. Unzip/untar the folder to generate 'tasks_1-20_v1-2' folder in current directory
3. Use dataproc.py to generate pickle files
4. Or directly use the pickle files
5. Run trainer.py to initialize the model, commence training, print graphs and report the mean accuracy.
6. Notebook files have also been provided for ease of access.


