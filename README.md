# **CATBDLs : Continuous Archimedean T-norm Based Differentiable Logics**

## **Project Overview**

This project  merges between logical reasoning
and deep learning by using only the t-norm generator to convert any prior knowledge expressed by first-order logic formulas into loss functions optimized
in TensorFlow..
## **Citeseer Experiments**
See the project for the supervised CiteSeer experiments. The CiteSeer data can be found in ``` datasets ``` folder.

## **Setup and Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/MohamedHssini/CATBDLs.git
   cd CATBDLs
   ```

2. **Create a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up the environment variables:**

   - Use the `.env` file in the project root and add your parameters:
     ```ini
     # Environment Variables
     EPOCHS = 200                   #Number of epochs for training the model.
     NUMBER_OF_SAMPLES = 10         #Number of random samples of training data emplyed.
     TYPE_GENERATOR_LIST = Yager,Acal    #Type of the t-norm generator used. Yager  for yager t-norms and Acal for Aczel-Alsina t-norms.
     SPLIT_LIST = 0.1, 0.25, 0.50   #Values of percentage employed  for spliting data into train and test data.
     PARAMETER_LIST_YAGER = 0.1,0.8,1.,10.  #Values taken to generate different Yager t-norms.
     PARAMETER_LIST_ACAL = 0.01,0.1,0.5,1.  #Values taken to generate different Aczel-Alsina t-norms.
     LEARNING_RATE_LIST = 0.0001
     ```


5. **Run the Project:**

   ```bash
   python CiteSeer_Experiment_new_22.py
   ```
