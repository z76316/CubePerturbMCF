# A Multi-step-ahead Markov Conditional Forward Model with Cube Perturbations for Extreme Weather Forecasting

Requirement
---
```
torch==1.4.0
torchfile==0.1.0
torchtext==0.4.0
torchvision==0.5.0
numpy==1.17.2
tqdm==4.42.0
sklearn==0.23.1
jupyter==1.0.0
jupyter-client==5.3.4
jupyter-console==6.0.0
jupyter-core==4.6.0
```

Data Preparation
---
The ExtremeWeather dataset can be downloaded from [here](https://academictorrents.com/details/c5bf370a90cae548d5a306c1be7d79186b9f60b9)

Preprocessing
---
Use ```data_preprocess_optimal_area.ipynb``` to preprocess:
1. In the step "2 Read Data", change the path of the data downloaded from previous step.
2. Change the ```label_type``` in the step "3. Get labels of original image ready for optimal area and time finding" to your target extreme weather event.
3. Set the path of the output folder in the step "7. Saving Preprocessed Data"
4. Then, go through all the preprocessing steps for the experiments input.

Experiments
---
Use ```cnn_exp.py``` to conduct experiment:
1. Change the ```folder_name``` and ```validation_folder_name``` in the ```cnn_exp.py``` to your preprocessed data.
2. Modifiy the output path at the end of the code. Or you can directly use the default output path.
3.
```shell=bash
python3 cnn_exp.py {perturbation rate} {perturbation type} {cube size}
