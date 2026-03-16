import requests
import json
print("Attempting invalid chat request to get the 500 error...")
# 1. First trigger an upload to create a session ID
files = {'files': ('empty.pdf', b'%PDF-1.4\n1 0 obj\n<</Type /Catalog>>\nendobj\n2 0 obj\n<</Type /Pages /Count 1 /Kids [3 0 R]>>\nendobj\n3 0 obj\n<</Type /Page /Parent 2 0 R /Resources <<>> /MediaBox [0 0 612 792]>>\nendobj\nxref\n0 4\n0000000000 65535 f \n0000000009 00000 n \n0000000052 00000 n \n0000000114 00000 n \ntrailer\n<</Size 4 /Root 1 0 R>>\nstartxref\n208\n%%EOF', 'application/pdf')}
data = {'deep_visual_mode': 'false'}
r = requests.post('http://localhost:3000/api/upload', files=files, data=data)

if r.status_code == 200:
    session_id = r.json().get('session_id')
    chat_data = {'session_id': session_id, 'message': 'What is this document about?'}
    r2 = requests.post('http://localhost:3000/api/chat', json=chat_data)
    print("CHAT:", r2.status_code, r2.text)
else:
    print("UPLOAD FAILED:", r.status_code, r.text)
