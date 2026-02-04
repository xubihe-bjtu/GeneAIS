# GenAISF

This is a PyTorch implementation of the paper: **GenAISF: A Generalizable Alignment-Interaction Framework for Station Weather Forecasting with Foundation Models.**

## Requirements

The model is implemented using **Python3.9** with dependencies specified in requirements.txt

## Usage

1. Install Python 3.9. For convenience, execute the following command.

~~~
pip install -r requirements.txt
~~~

2. **Prepare Data**: You can obtain the three well-preprocessed datasets—`ChinaNorth`, `ChinaSouth`, and `US_r3`—from the following [Google Drive](https://drive.google.com/drive/folders/1cusG3muIFew5c1FLZWWI-O4cZHNBLI4H). After downloading, place the datasets in the `./dataset` folder.
3. To train and evaluate the model, simply execute the following examples within the `GenAISF/Runner` directory. The `target` parameter specifies the weather variable of interest: 0 for U-speed, 1 for V-speed, 2 for MSL, and 3 for TMP.

~~~
#ChinaNorth
cd ChinaNorth
python run.py --model GenAISF --target 0

#ChinaSouth
cd ChinaSouth
python run.py --model GenAISF --target 0

#US_r3
cd US_r3
python run.py --model GenAISF --target 0
~~~

