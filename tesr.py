import google.generativeai as genai

genai.configure(api_key="AIzaSyCYJjUlC3LXkr9m_ebdjkwPkey3KlrsHA8")
model = genai.GenerativeModel('models/gemini-1.5-pro-latest')

response = model.generate_content("Hello Gemini! What can you do?")
print(response.text)

