import torch
import gradio as gr
import matplotlib.pyplot as plt

model = torch.load('python/ai/model.pkl')


def get_predition(Re: float,Im: float) -> list[float]:
    X_test = torch.tensor([Re, Im])
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = torch.clamp(y_pred,0,1)
    return y_pred.numpy()[0]


input1 = gr.Slider(minimum=-0.999, maximum=0.999, label="Real")
input2 = gr.Slider(minimum=-0.999, maximum=0.999, label="Imaginary")

output = gr.Image(type="numpy", image_mode='L')


gr.Interface(fn=get_predition,
             inputs=[input1,input2],
             outputs=output).launch()