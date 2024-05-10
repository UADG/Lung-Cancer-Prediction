# !pip install gradio ipywidgets
import pandas as pd
import gradio as gr
import joblib
gnb_joblib = joblib.load("pipeline.joblib")
scaler_joblib = joblib.load("scaler.joblib")

def predict(Balanced_Diet, Wheezing, Alcohol_Use, Obesity, Fatigue, Clubbing_of_Finger_Nails, Coughing_of_Blood, Air_Pollution, Swallowing_Difficulty, Frequent_Cold, Snoring):
    sample = pd.DataFrame({
        "Balanced Diet": [Balanced_Diet],
        "Wheezing": [Wheezing],
        "Alcohol use": [Alcohol_Use],
        "Obesity": [Obesity],
        "Fatigue": [Fatigue],
        "Clubbing of Finger Nails": [Clubbing_of_Finger_Nails],
        "Coughing of Blood": [Coughing_of_Blood],
        
        "Air Pollution": [Air_Pollution],
        "Swallowing Difficulty": [Swallowing_Difficulty],
        "Frequent Cold": [Frequent_Cold],
        "Snoring": [Snoring]
    })
    # แก้ชื่อฟีเจอร์ให้เป็นลำดับเดียวกับที่ใช้ในการฝึกโมเดล
    sample = sample[['Air Pollution', 'Alcohol use', 'Balanced Diet', 'Obesity',
                     'Coughing of Blood', 'Fatigue', 'Wheezing', 'Swallowing Difficulty',
                     'Clubbing of Finger Nails', 'Frequent Cold', 'Snoring']]
    sample = scaler_joblib.transfrom(sample)
    sample = [i+1.6 for i in sample]
    sample = pd.DataFrame(sample)
    cancer = pipeline.predict(sample)
    
    return str(cancer[0])

# https://www.gradio.app/guides
with gr.Blocks() as blocks:
    Air_Pollution = gr.Slider(label="Air_Pollution", minimum=0, maximum=10, step=1)
    Alcohol_Use = gr.Slider(label="Alcohol_Use", minimum=0, maximum=10, step=1)
    Balanced_Diet = gr.Slider(label="Balanced_Diet", minimum=0, maximum=10, step=1)
    Obesity = gr.Slider(label="Obesity", minimum=0, maximum=10, step=1)
    Coughing_of_Blood = gr.Slider(label="Coughing_of_Blood", minimum=0, maximum=10, step=1)
    Fatigue = gr.Slider(label="Fatigue", minimum=0, maximum=10, step=1)
    Wheezing = gr.Slider(label="Wheezing", minimum=0, maximum=10, step=1)
    Swallowing_Difficulty = gr.Slider(label="Swallowing_Difficulty", minimum=0, maximum=10, step=1)
    Clubbing_of_Finger_Nails = gr.Slider(label="Clubbing_of_Finger_Nails", minimum=0, maximum=10, step=1)
    Frequent_Cold = gr.Slider(label="Frequent_Cold", minimum=0, maximum=10, step=1)
    Snoring = gr.Slider(label="Snoring", minimum=0, maximum=10, step=1)

    
    cancer = gr.Textbox(label="Level")

    inputs = [Balanced_Diet, Wheezing, Alcohol_Use, Obesity, Fatigue, Clubbing_of_Finger_Nails, Coughing_of_Blood, Air_Pollution, Swallowing_Difficulty, Frequent_Cold, Snoring]
    outputs = [cancer]

    predict_btn = gr.Button("Predict")
    predict_btn.click(predict, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    blocks.launch() # Local machine only
    # blocks.launch(server_name="0.0.0.0") # LAN access to local machine
    # blocks.launch(share=True) # Public access to local machine
