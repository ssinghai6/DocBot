import requests
files = {'files': ('test.pdf', open('Sanshrit_Singhai_Resume.pdf', 'rb'), 'application/pdf')}
data = {'deep_visual_mode': 'false'}
r = requests.post('http://localhost:3000/api/upload', files=files, data=data)
print("UPLOAD:", r.status_code)
if r.status_code == 200:
    session_id = r.json().get('session_id')
    chat_data = {'session_id': session_id, 'message': 'What is this document about?'}
    r2 = requests.post('http://localhost:3000/api/chat', json=chat_data)
    print("CHAT:", r2.status_code)
