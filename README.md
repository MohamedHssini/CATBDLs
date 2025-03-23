# **CiteSeer Experiment**

## **Project Overview**

This project is a **aaaaaaaa** .

## **Setup and Installation**

1. **Clone the repository:**

   ```bash
   git clone https://github.com/...
   cd project
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

   - Create a `.env` file in the project root and add your parameters:
     ```ini
     # Environment Variables
     EPOCHS = 300
     NUMBER_OF_SAMPLES = 1
     TYPE_GENERATOR_LIST = Yager,Acal
     SPLIT_LIST = 0.1, 0.25, 0.50, 0.75, 0.90
     PARAMETER_LIST_YAGER = 0.1,0.8,1.,10.,50.,100.,150.
     PARAMETER_LIST_ACAL = 0.01,0.1,0.5,1.,10.,26.7,30.
     LEARNING_RATE_LIST = 0.0001
     ```


5. **Run the Project:**

   ```bash
   python CiteSeer_Experiment_new.py
   ```
