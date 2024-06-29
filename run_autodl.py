jobs = [
    'python3 tries/t005_from004_longer_epoch.py'
    ]

import os
import sys
import requests
import subprocess
os.environ['HF_HOME'] = '/root/autodl-tmp/cache/'

for job in jobs:
    msg_sent = 0
    try:
        subprocess.check_call(job, shell=True)
    except:
        print('!!!!!!!!!!!!!Failed!!!!!!!!!!!!')
        headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjQyNjMwMCwidXVpZCI6ImNjYmRmZWYyLWI4ODQtNDM5YS1hNmM2LThjM2M2MGNiY2FmMiIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.JCFY0Rt9JWpRFDIOzlY86V7jhKZFCvJ2Hrjy0HVKhw3v1ULOsNhAhVhvEzyYUEkxZ_lhgYYHd6LpGEmTRi-4BA"}
        resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                            json={
                                "title": f"实验失败{job}",
                                "name": f"实验失败{job}",
                                "content": "bad"
                            }, headers = headers)
        print(resp.content.decode())
        msg_sent = 1
    if msg_sent == 0:
        headers = {"Authorization": "eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjQyNjMwMCwidXVpZCI6ImNjYmRmZWYyLWI4ODQtNDM5YS1hNmM2LThjM2M2MGNiY2FmMiIsImlzX2FkbWluIjpmYWxzZSwiYmFja3N0YWdlX3JvbGUiOiIiLCJpc19zdXBlcl9hZG1pbiI6ZmFsc2UsInN1Yl9uYW1lIjoiIiwidGVuYW50IjoiYXV0b2RsIiwidXBrIjoiIn0.JCFY0Rt9JWpRFDIOzlY86V7jhKZFCvJ2Hrjy0HVKhw3v1ULOsNhAhVhvEzyYUEkxZ_lhgYYHd6LpGEmTRi-4BA"}
        resp = requests.post("https://www.autodl.com/api/v1/wechat/message/send",
                            json={
                                "title": f"实验成功{job}",
                                "name": f"实验成功{job}",
                                "content": "good"
                            }, headers = headers)
        print(resp.content.decode())
os.system("/usr/bin/shutdown")