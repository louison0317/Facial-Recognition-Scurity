from __future__ import unicode_literals
import csv
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
from flask import Flask,render_template,request,flash,abort
from flask_sqlalchemy import SQLAlchemy
import sqlite3 as sql
import pymysql
from skimage import io
import dlib
#import win32api
import os           # 读写文件
import shutil
import numpy as np
pymysql.install_as_MySQLdb()

app = Flask(__name__)
DATABASE_URI = "member_local_line.sqlite"
# LINE 聊天機器人的基本資料(這裡要改成自己的)
line_bot_api = LineBotApi('')
handler = WebhookHandler('')


# 接收 LINE 的資訊
@app.route("/callback", methods=['POST'])
def callback():
    signature = request.headers['X-Line-Signature']

    body = request.get_data(as_text=True)
    app.logger.info("Request body: " + body)

    try:
        handler.handle(body, signature)
    except InvalidSignatureError:
        abort(400)

    return 'OK'


# 學你說話
@handler.add(MessageEvent, message=TextMessage)
def echo(event):

    text = event.message.text.split( )

    if text[0] =="綁定":
        account = text[1]
        secret = text[2]
        line = event.source.user_id
        account_list=[]
        secret_list =[]
        line_list=[]
        account_now = "'"+account+"'"
        secret_now = "'"+secret+"'"
        line_now = "'"+event.source.user_id+"'"
        #line_bot_api.push_message(event.source.user_id, TextSendMessage(text="成功拉XD"))
        with sql.connect(DATABASE_URI) as con:
            cur = con.cursor()

            sqlstr = 'SELECT * FROM member'
            cur.execute(sqlstr)
            rows = cur.fetchall()
            for row in rows:

                account_list.append(row[0])
                secret_list.append(row[1])
                line_list.append(row[7])

            cur.execute("update member set line = "+line_now+ " where account = "+account_now +"and secret = "+secret_now)
            con.commit()
        account_check = False
        secret_check = False
        exist_check = False
        for i in range(len(account_list)):
            if account_list[i] == account:
                account_check = True
                if secret_list[i] == secret:
                    secret_check =True
                    break
                else:
                    break
        for i in range(len(line_list)):
            if line_list[i] == line:
                exist_check = True
        if exist_check == True:
            line_bot_api.reply_message(
                event.reply_token,
                TextSendMessage(text='此line以綁定其他帳號')
            )
        else:
            if account_check == False:

                line_bot_api.reply_message(
                    event.reply_token,
                    TextSendMessage(text= '帳號不存在')
            )
            else:
                if secret_check == False:
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text= '密碼不正確')
                )
                else:
                    line_bot_api.reply_message(
                        event.reply_token,
                        TextSendMessage(text='綁定成功')
                    )
    elif text[0] == "指令":
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='以下指令都請用「空格」隔開 \n\n'      
                            '『綁定 (帳號名) (密碼) 』\nex. 綁定 andy123 123456 \n =>來綁定line到帳戶裡\n\n'
                            '『查詢聯絡人』 \n =>查詢帳戶綁定的聯絡人\n\n '
                            '『新增聯絡人』 \nex.新增聯絡人 andy\nex.新增聯絡人 andy,cindy\n=>新增帳戶綁定的聯絡人\n(若為多位聯絡人請用,隔開)\n\n'
                            '『刪除聯絡人』 \nex.刪除聯絡人 andy\nex.刪除聯絡人 andy,cindy\n=>刪除帳戶綁定的聯絡人\n(若為多位聯絡人請用,隔開)\n\n'
                            '『指令』 \n=>查詢所有指令\n\n')
        )
    elif text[0] == "新增聯絡人":
        new_relation = text[1]
        line_now = event.source.user_id
        line_final = "'" + event.source.user_id + "'"
        line_list=[]
        relation_list =[]
        with sql.connect(DATABASE_URI) as con:
            cur = con.cursor()
            sqlstr = 'SELECT * FROM member'
            cur.execute(sqlstr)
            rows = cur.fetchall()
            for row in rows:
                line_list.append(row[7])
                relation_list.append(row[4])

            index = line_list.index(line_now)
            if relation_list[index]=="":
                final_relation = "'"+new_relation+"'"
            else:
                final_relation = "'" + relation_list[index] + ',' + new_relation + "'"
            cur.execute(
                "update member set relation = " + final_relation + " where line = " + line_final)
            con.commit()
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='新增聯絡人成功!!!目前聯絡人為'+final_relation)
        )
    elif text[0] == "刪除聯絡人":

        delete_list = text[1].split(',')
        print(delete_list)
        line_now = event.source.user_id
        line_final = "'" + event.source.user_id + "'"
        line_list=[]
        relation_list =[]
        with sql.connect(DATABASE_URI) as con:
            cur = con.cursor()
            sqlstr = 'SELECT * FROM member'
            cur.execute(sqlstr)
            rows = cur.fetchall()
            for row in rows:
                line_list.append(row[7])
                relation_list.append(row[4])

            index = line_list.index(line_now)
            final_relation=""
            print(relation_list[index])
            old_list = relation_list[index].split(',')
            print(old_list)
            for i in range(len(old_list)):
                count = 0
                for j in range(len(delete_list)):
                    if old_list[i] != delete_list[j]:
                        count+=1
                if count == len(delete_list):
                    if final_relation=="":
                        final_relation = final_relation + old_list[i]
                    else:
                        final_relation = final_relation +","+old_list[i]
            print(final_relation)
            final_relation = "'"+final_relation+"'"
            cur.execute(
                "update member set relation = " + final_relation + " where line = " + line_final)
            con.commit()
            x = final_relation
            print(x)
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='刪除聯絡人成功!!!目前聯絡人為'+x)
        )
    elif text[0] == "查詢聯絡人":
        line_now = event.source.user_id
        line_list = []
        relation_list = []
        with sql.connect(DATABASE_URI) as con:
            cur = con.cursor()
            sqlstr = 'SELECT * FROM member'
            cur.execute(sqlstr)
            rows = cur.fetchall()
            for row in rows:
                line_list.append(row[7])
                relation_list.append(row[4])
            index = line_list.index(line_now)
            text = relation_list[index]
            con.commit()
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='目前聯絡人為'+text)
        )
    else:
        line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text='無法辨識，請確認指令!\n輸入「指令」來查詢指令')
        )

# 1. Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# 2. Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# 3. Dlib Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")

# 返回单张图像的 128D 特征
def return_128d_features(path_img):
    img_rd = io.imread(path_img)
    faces = detector(img_rd, 1)

    print("%-40s %-20s" % ("检测到人脸的图像 / Image with faces detected:", path_img), '\n')

    # 因为有可能截下来的人脸再去检测，检测不出来人脸了
    # 所以要确保是 检测到人脸的人脸图像 拿去算特征
    if len(faces) != 0:
        shape = predictor(img_rd, faces[0])
        face_descriptor = face_reco_model.compute_face_descriptor(img_rd, shape)
    else:
        face_descriptor = 0
        print("no face")
    print(face_descriptor)
    return face_descriptor



## 注册
@app.route('/',methods=['GET','POST'])
def register():
    if request.method == 'GET':
        return render_template('index.html')
    else:

        msg = ""
        try:
            account_list =[]
            account_check =False
            account = request.form['inputAccount']
            password = request.form['inputPassword']
            name = request.form['inputName']
            type = request.form['inputType']
            relation = request.form['inputRelation']
            image = request.files['image']


            with sql.connect(DATABASE_URI) as con:
                cur = con.cursor()
                sqlstr = 'SELECT * FROM member'
                cur.execute(sqlstr)
                rows = cur.fetchall()
                for row in rows:
                    account_list.append(row[0])
                for i in range(len(account_list)):
                    if account_list[i] == account:
                        msg = "帳號名稱已被註冊!請使用其他名稱!"
                        print("帳號名稱已被註冊")
                        account_check = True

                        break
                if account_check == False:
                    image.save('C:/Users/USER/PycharmProjects/account_flask/memberPic/{}.jpg'.format(account))
                    picfile = account + '.jpg'
                    print('?')
                    print(account, password, name, type, relation)
                    f = return_128d_features('memberPic/' + account + '.jpg')

                    # print(f+"=====================")
                    # if f == 0:
                    #     print("沒有檢測到人臉")

                    with open('C:/Users/USER/PycharmProjects/account_flask/myfeature/' + account + '.csv', "w",
                              newline="") as csvfile:
                        writer = csv.writer(csvfile)

                        writer.writerow(f)

                        f_file = account + '.csv'
                    cur.execute("INSERT INTO member(account, secret,name,type,relation,picture,feature) VALUES (?,?,?,?,?,?,?)",
                             (account, password,name,type,relation,picfile,f_file))
                    msg = "註冊成功！歡迎加入校園守護天使!"
                con.commit()
                print("yes")


        except:
            print('HI')
            msg = "沒有上傳圖片或偵測不到人臉!註冊失敗！"
        finally:

            return render_template('index.html', msg=msg)
@app.route('/image',methods=['GET','POST'])
def image():
    if request.method == 'GET':
        return render_template('image.html')
    else:

        img = request.files['image']

        img.save('C:/Users/USER/PycharmProjects/account_flask/memberPic/{}.jpg'.format(123))
        return render_template('image.html')
if __name__ == '__main__':
    app.run(debug=True)
