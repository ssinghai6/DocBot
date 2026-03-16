import subprocess
try:
    subprocess.run(["python3", "test_empty_chat.py"])
except Exception as e:
    print(e)
