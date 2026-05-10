import google.generativeai as genai

genai.configure(api_key="AIzaSyBwSnN93it6K2EYyvoJKQzQO-J-lsm-AGg")

models = genai.list_models()

for m in models:
    print(m.name)