import glob
import os

import yagmail

def auto_send_emails(to, subject, contents, attachments=None):
    # 链接邮箱服务器
    yag = yagmail.SMTP(user="*********@qq.com", password="*********", host='smtp.exmail.qq.com')
    # 给多个用户发送带附件邮件
    yag.send(to=to, subject=subject, contents=contents)