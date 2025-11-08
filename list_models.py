import google.generativeai as genai

genai.configure(api_key="AIzaSyBrUfmyKFDyNOYbNzMZCqXoie1_SAS4h0E")

models = genai.list_models()
for m in models:
    print(m.name)