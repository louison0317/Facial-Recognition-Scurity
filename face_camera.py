# 摄像头实时人脸识别
# Real-time face recognition
from datetime import datetime

import sqlite3
import dlib         # 人脸处理的库 Dlib
import numpy as np  # 数据处理的库 Numpy
import cv2          # 图像处理的库 OpenCV
import pandas as pd # 数据处理的库 Pandas
import os
import time
from PIL import Image, ImageDraw, ImageFont
from linebot import LineBotApi, WebhookHandler
from linebot.exceptions import InvalidSignatureError
from linebot.models import MessageEvent, TextMessage, TextSendMessage
# LINE 聊天機器人的基本資料
line_bot_api = LineBotApi('')
handler = WebhookHandler('')

# 1. Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()

# 2. Dlib 人脸 landmark 特征点检测器
predictor = dlib.shape_predictor('data/data_dlib/shape_predictor_68_face_landmarks.dat')

# 3. Dlib Resnet 人脸识别模型，提取 128D 的特征矢量
face_reco_model = dlib.face_recognition_model_v1("data/data_dlib/dlib_face_recognition_resnet_model_v1.dat")


class Face_Recognizer:
    def __init__(self):
        #這裡填入老師的line
        self.teacher_line="老師的line"
        self.teacher_name="老師的account name"
        # 用来存放所有录入人脸特征的数组 / Save the features of faces in the database
        self.features_known_list = []

        # 存储录入人脸名字 / Save the name of faces known
        self.name_known_cnt = 0
        self.name_known_list = []

        # 存储当前摄像头中捕获到的所有人脸的坐标名字 / Save the positions and names of current faces captured
        self.pos_camera_list = []
        self.name_camera_list = []
        # 存储当前摄像头中捕获到的人脸数
        self.faces_cnt = 0
        # 存储当前摄像头中捕获到的人脸特征
        self.features_camera_list = []

        # Update FPS
        self.fps = 0
        self.frame_start_time = 0

        #陌生人數
        self.stranger_count =0
        #家長人數
        self.parent_count =0
        # 學生人數
        self.student_count = 0
        #老師人數
        self.teacher_count =0
        #會員資料
        self.account_list=[]
        self.name_list = []
        self.type_list = []
        self.relation_list = []
        self.line_list = []

    def get_child(self,index):
        name_list = []
        child = self.relation_list[index].split(',')
        if self.teacher_name in child:
            child.remove(self.teacher_name)
        for i in child:
            name_list.append(self.name_list[self.account_list.index(i)])

        return name_list
    def get_parent_line(self,index):


        call_line_list = []
        n = self.relation_list[index]
        x = n.split(',')

        for i in range(len(x)):
            index_line = self.account_list.index(x[i])

            call_line_list.append(self.line_list[index_line])

        return call_line_list
    def get_in_out(self,time_want):
        index_s = time_want.index('s')
        index_m = time_want.index('m')
        index_h = time_want.index('h')
        digits_second = time_want[index_s - 1]
        tens_second = time_want[index_s - 2]
        digits_minute = time_want[index_m - 1]
        tens_minute = time_want[index_m - 2]
        digits_hour =time_want[index_h - 1]
        tens_hour = time_want[index_h - 2]
        # print((int)(tens_hour + digits_hour))
        # print((int)(tens_minute + digits_minute))
        # print((int)(tens_second + digits_second))
        h = (int)(tens_hour + digits_hour)
        m = (int)(tens_minute + digits_minute)
        s = (int)(tens_second + digits_second)
        time = (h * 60 * 60) + (m * 60) + s
        return time
    def get_last_record(self, find):
        last_index =0
        if len(self.record_account_list) ==0:
            return last_index
        else:
            for i in range(len(self.record_account_list)):
                # print("================")
                # print(self.record_account_list[i])
                # print(find)
                # print("================")
                if self.record_account_list[i] == find:
                    # print("進來了")
                    last_index = i
        return last_index

    # 从 "features_all.csv" 读取录入人脸特征
    def get_face_database(self):
        conn = sqlite3.connect('member_local_line.sqlite')
        cursor = conn.cursor()
        sqlstr = 'SELECT * FROM member'
        cursor.execute(sqlstr)
        rows = cursor.fetchall()
        for row in rows:
            if os.path.exists('myfeature/'+row[0] +'.csv'):
                path_features_known_csv = 'myfeature/'+row[0]+'.csv'
                csv_rd = pd.read_csv(path_features_known_csv, header=None)
                # 2. 读取已知人脸数据 / Print known faces
                for i in range(csv_rd.shape[0]):
                    features_someone_arr = []
                    for j in range(0, 128):
                        if csv_rd.iloc[i][j] == '':
                            features_someone_arr.append('0')
                        else:
                            features_someone_arr.append(csv_rd.iloc[i][j])
                    self.features_known_list.append(features_someone_arr)
                    self.name_known_cnt = len(self.name_known_list)
                print("Faces in Database：", len(self.features_known_list))


            else:
                print('##### Warning #####', '\n')
                print(" no data in database!")
                print(
                "Please run 'camera.py' to build account",
                '\n')
                print('##### End Warning #####')
                return 0
        print(self.account_list)
        print(self.name_list)
        print(self.type_list)
        print(self.relation_list)
        print(self.line_list)
        return 1

    # 计算两个128D向量间的欧式距离 / Compute the e-distance between two 128D features
    @staticmethod
    def return_euclidean_distance(feature_1, feature_2):
        feature_1 = np.array(feature_1)
        feature_2 = np.array(feature_2)
        dist = np.sqrt(np.sum(np.square(feature_1 - feature_2)))
        return dist

    # 更新 FPS / Update FPS of Video stream
    def update_fps(self):
        now = time.time()
        self.frame_time = now - self.frame_start_time
        self.fps = 1.0 / self.frame_time
        self.frame_start_time = now

    def draw_note(self, img_rd):
        font = cv2.FONT_ITALIC

        #cv2.putText(img_rd, "Face Recognizer", (20, 40), font, 1, (255, 255, 255), 1, cv2.LINE_AA)
        #cv2.putText(img_rd, "FPS:   " + str(self.fps.__round__(2)), (20, 100), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Faces: " + str(self.faces_cnt), (20, 140), font, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.putText(img_rd, "Q: Quit", (20, 450), font, 0.8, (255, 255, 255), 1, cv2.LINE_AA)

    def draw_name(self, img_rd):
        time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # 檔案名稱不能有 ":"，所以改h、m、s
        file_time = datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss')
        faces = detector(img_rd, 0)
        # 在人脸框下面写人脸名字 / Write names under rectangle
        font = ImageFont.truetype("C:\Windows\Fonts\simsun.ttc", 30)
        img = Image.fromarray(cv2.cvtColor(img_rd, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        for i in range(self.faces_cnt):
            # cv2.putText(img_rd, self.name_camera_list[i], self.pos_camera_list[i], font, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
            if self.name_camera_list[i] =='unknown':
                draw.text(xy=self.pos_camera_list[i], text='陌生人', font=font)
                img_with_name = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
            else:
                index = self.account_list.index(self.name_camera_list[i])
                con = sqlite3.connect('member_local_line.sqlite')
                last_record = self.get_last_record(self.account_list[index])
                if 'unknown' not in self.name_camera_list:
                    self.stranger_count =0
                if self.type_list[index] == "家長":
                    self.parent_count +=1
                if self.type_list[index] == "學生":
                    self.student_count +=1
                if self.type_list[index] =="老師":
                    self.teacher_count +=1
                if self.parent_count > 3:
                    self.parent_count = 0
                if self.student_count > 3:
                    self.student_count = 0
                if self.teacher_count > 3:
                    self.teacher_count = 0
                if self.stranger_count > 3:
                    self.stranger_count = 0
                print("學生數"+(str)(self.student_count))
                print("家長數" + (str)(self.parent_count))
                print("老師數" + (str)(self.teacher_count))
                print("陌生人數" + (str)(self.stranger_count))
                if self.parent_count > 2:
                    if self.type_list[index] == "家長":
                        child_name=""
                        child_name_list =self.get_child(index)
                        for word in child_name_list:
                            child_name += word
                        # 沒有記錄過就插入
                        if last_record == 0:
                            cursorObj = con.cursor()
                            cursorObj.execute(
                                "INSERT INTO record_easy(type,name,account,in_time,out_time,strangerPic) VALUES (?,?,?,?,?,?)",
                                (self.type_list[index], self.name_list[index], self.account_list[index], file_time,
                                    None, None))
                            con.commit()
                            if len(child_name)>0 :
                                print("老師!家長(" + self.name_list[index] + ")來接小孩(" + child_name + ")了!")
                                line_bot_api.push_message(self.teacher_line, TextSendMessage(text="老師!家長(" + self.name_list[index] + ")來接小孩(" + child_name + ")了!"))
                                self.parent_count = 0
                            else:
                                print("老師!家長(" + self.name_list[index] + ")來接小孩了!")
                                line_bot_api.push_message(self.teacher_line, TextSendMessage(text="老師!家長(" + self.name_list[index] + ")來接小孩了!"))

                                self.parent_count = 0
                        else:
                            check_in = self.record_in_time_list[last_record]
                            check_out = self.record_out_time_list[last_record]
                                # 相差十秒才紀錄，不然一直增加
                                # print("時間拉拉拉拉拉拉拉拉拉拉阿")
                                # print(self.get_in_out(check))
                                # print(self.get_in_out(file_time))

                            if (self.get_in_out(file_time) - self.get_in_out(check_in)) > 5:
                                    # 有記錄過但沒有"出來"，就修改加上out_time
                                if self.record_out_time_list[last_record] == None:
                                    x = "'" + self.record_in_time_list[last_record] + "'"
                                    y = "'" + self.account_list[index] + "'"
                                    cursorObj = con.cursor()
                                    cursorObj.execute(
                                            "UPDATE record_easy SET out_time = " + "'" + file_time + "'" + " WHERE in_time = " + x + "AND account = " + y)
                                    con.commit()
                                    print("老師!家長(" + self.name_list[index] + ")離開了!")
                                    line_bot_api.push_message(self.teacher_line, TextSendMessage(text="老師!家長(" + self.name_list[index] + ")離開了!"))
                                    self.parent_count = 0
                                    # 剩下的就是新進來的，就新建一行
                                else:
                                    if (self.get_in_out(file_time) - self.get_in_out(check_out)) > 5:
                                        print(self.get_in_out(file_time))
                                        print(self.get_in_out(check_out))
                                        print((self.get_in_out(file_time) - self.get_in_out(check_out)))
                                        cursorObj = con.cursor()
                                        cursorObj.execute(
                                                "INSERT INTO record_easy(type,name,account,in_time,out_time,strangerPic) VALUES (?,?,?,?,?,?)",
                                            (self.type_list[index], self.name_list[index], self.account_list[index],
                                                 file_time,
                                                 None, None))
                                        con.commit()
                                        if len(child_name) > 0:

                                            print("老師!家長(" + self.name_list[index] + ")來接小孩(" + child_name + ")了!")
                                            line_bot_api.push_message(self.teacher_line, TextSendMessage(text="老師!家長(" + self.name_list[index] + ")來接小孩(" + child_name + ")了!"))
                                            self.parent_count = 0
                                        else:

                                            print("老師!家長(" + self.name_list[index] + ")來接小孩了!")
                                            line_bot_api.push_message(self.teacher_line, TextSendMessage(text="老師!家長(" + self.name_list[index] + ")來接小孩了!"))
                                            self.parent_count = 0



                if self.student_count > 2:
                    # 沒有記錄過就插入
                    if self.type_list[index] == "學生":
                        if last_record == 0:
                            cursorObj = con.cursor()
                            cursorObj.execute(
                                "INSERT INTO record_easy(type,name,account,in_time,out_time,strangerPic) VALUES (?,?,?,?,?,?)",
                                (self.type_list[index], self.name_list[index], self.account_list[index], file_time, None,
                                 None))
                            con.commit()
                            print("老師!學生(" + self.name_list[index] + ")到學校了!")
                            self.student_count = 0
                            line_bot_api.push_message(self.teacher_line, TextSendMessage(text="老師!學生(" + self.name_list[index] + ")到學校了!"))
                        else:
                            check_in = self.record_in_time_list[last_record]
                            check_out = self.record_out_time_list[last_record]
                            # 相差十秒才紀錄，不然一直增加
                            # print("時間拉拉拉拉拉拉拉拉拉拉阿")
                            # print(self.get_in_out(check))
                            # print(self.get_in_out(file_time))

                            if (self.get_in_out(file_time) - self.get_in_out(check_in)) > 5:
                                # 有記錄過但沒有出來，就修改加上out_time
                                if self.record_out_time_list[last_record] == None:
                                    x = "'" + self.record_in_time_list[last_record] + "'"
                                    y = "'" + self.account_list[index] + "'"
                                    cursorObj = con.cursor()
                                    cursorObj.execute(
                                        "UPDATE record_easy SET out_time = " + "'" + file_time + "'" + " WHERE in_time = " + x + "AND account = " + y)
                                    con.commit()

                                    print("老師!學生(" + self.name_list[index] + ")離開學校了!")
                                    self.student_count = 0
                                    line_bot_api.push_message(self.teacher_line, TextSendMessage(text="老師!學生(" + self.name_list[index] + ")離開學校了!"))

                                # 剩下的就是新進來的，就新建一行
                                else:
                                    if (self.get_in_out(file_time) - self.get_in_out(check_out)) > 5:
                                        print(self.get_in_out(file_time))
                                        print(self.get_in_out(check_out))
                                        print((self.get_in_out(file_time) - self.get_in_out(check_out)))
                                        cursorObj = con.cursor()
                                        cursorObj.execute(
                                            "INSERT INTO record_easy(type,name,account,in_time,out_time,strangerPic) VALUES (?,?,?,?,?,?)",
                                            (self.type_list[index], self.name_list[index], self.account_list[index],
                                             file_time,
                                             None, None))
                                        con.commit()
                                        people_in = True

                                        print("老師!學生(" + self.name_list[index] + ")到學校了!")
                                        self.student_count = 0
                                        line_bot_api.push_message(self.teacher_line, TextSendMessage(text="老師!學生(" + self.name_list[index] + ")到學校了!"))
                if self.teacher_count >2:
                    # 沒有記錄過就插入
                    if self.type_list[index] == "老師":
                        if last_record == 0:
                            cursorObj = con.cursor()
                            cursorObj.execute(
                                "INSERT INTO record_easy(type,name,account,in_time,out_time,strangerPic) VALUES (?,?,?,?,?,?)",
                                (self.type_list[index], self.name_list[index], self.account_list[index], file_time, None,
                                 None))
                            con.commit()
                            print("老師到學校了!")
                            self.teacher_count = 0

                        else:
                            check_in = self.record_in_time_list[last_record]
                            check_out = self.record_out_time_list[last_record]
                            # 相差十秒才紀錄，不然一直增加
                            # print("時間拉拉拉拉拉拉拉拉拉拉阿")
                            # print(self.get_in_out(check))
                            # print(self.get_in_out(file_time))

                            if (self.get_in_out(file_time) - self.get_in_out(check_in)) > 5:
                                # 有記錄過但沒有出來，就修改加上out_time
                                if self.record_out_time_list[last_record] == None:
                                    x = "'" + self.record_in_time_list[last_record] + "'"
                                    y = "'" + self.account_list[index] + "'"
                                    cursorObj = con.cursor()
                                    cursorObj.execute(
                                        "UPDATE record_easy SET out_time = " + "'" + file_time + "'" + " WHERE in_time = " + x + "AND account = " + y)
                                    con.commit()

                                    print("老師離開學校了!")
                                    self.teacher_count = 0


                                # 剩下的就是新進來的，就新建一行
                                else:
                                    if (self.get_in_out(file_time) - self.get_in_out(check_out)) > 5:
                                        print(self.get_in_out(file_time))
                                        print(self.get_in_out(check_out))
                                        print((self.get_in_out(file_time) - self.get_in_out(check_out)))
                                        cursorObj = con.cursor()
                                        cursorObj.execute(
                                            "INSERT INTO record_easy(type,name,account,in_time,out_time,strangerPic) VALUES (?,?,?,?,?,?)",
                                            (self.type_list[index], self.name_list[index], self.account_list[index],
                                             file_time,
                                             None, None))
                                        con.commit()
                                        people_in = True

                                        print("老師到學校了!")
                                        self.teacher_count = 0


                draw.text(xy=self.pos_camera_list[i], text=self.name_list[index] ,font = font)
                img_with_name = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        # conn = sqlite3.connect('member_local_line.sqlite')
        # cursor = conn.cursor()



            self.know_count = 0
        if 'unknown' in  self.name_camera_list:
            text = "danger!!!!!"
            self.stranger_count += 1
            if self.stranger_count > 3:
                self.stranger_count = 0
            print(self.stranger_count)
            cv2.putText(img_with_name, text, (120, 70), cv2.FONT_HERSHEY_PLAIN, 6.0, (0, 0, 255), 2)
        print((6/self.faces_cnt))


        if self.stranger_count > 2:
            time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            #檔案名稱不能有 ":"，所以改h、m、s
            file_time = datetime.now().strftime('%Y-%m-%d %Hh%Mm%Ss')
            con = sqlite3.connect('member_local_line.sqlite')
            cursorObj = con.cursor()
            last_record = self.get_last_record('unknown')
            check_in = self.record_in_time_list[last_record]

            if (self.get_in_out(file_time) - self.get_in_out(check_in)) > 3:
                cursorObj.execute(
                    "INSERT INTO record_easy(type,name,account,in_time,out_time,strangerPic) VALUES (?,?,?,?,?,?)",
                    ("入侵", "陌生人", "unknown", file_time, None, file_time + '.jpg'))
                con.commit()
                student_check = self.name_camera_list
                if 'unknown' in student_check:
                    student_check.remove('unknown')
                if len(student_check)>0 and ('unknown' not in student_check):
                    for i in range(len(student_check)):
                        index = self.account_list.index(student_check[i])
                        if self.type_list[index] == "學生":
                            call_list = self.get_parent_line(index)
                            cv2.imwrite('strangers/' + file_time + '.jpg', img_rd)
                            print(call_list)
                            print("抓到你啦(帶學生)!!!!!!!!!!!!!!!!")
                            print("有陌生人在" + time + "進入校園!!!帶走" + self.name_list[index] + "了!!!")
                            for i in call_list:
                                line_bot_api.push_message(i, TextSendMessage(text="有陌生人在" + time + "進入校園!!!帶走" + self.name_list[index] + "了!!!"))
                            self.stranger_count = 0
                else:
                    index = self.account_list.index(self.teacher_name)
                    call_list =self.get_parent_line(index)
                    cv2.imwrite('strangers/'+file_time+'.jpg',img_rd)
                    print(call_list)
                    print("抓到你啦(一個人)!!!!!!!!!!!!!!!!")
                    for i in call_list:
                        line_bot_api.push_message(i, TextSendMessage(text="有陌生人在"+time+"進入校園!!!請保持警戒!!!"))
                self.stranger_count = 0
        return img_with_name



    # 处理获取的视频流，进行人脸识别 / Input video stream and face reco process
    def process(self, stream):
        conn = sqlite3.connect('member_local_line.sqlite')
        cursor = conn.cursor()
        sqlstr = 'SELECT * FROM member'
        cursor.execute(sqlstr)
        rows = cursor.fetchall()
        member_picture = []


        for row in rows:
            self.name_known_list.append(row[0])
        print(self.name_known_list)

        for row in rows:
            member_picture.append(row[5])
        print(member_picture)


        # 1. 读取存放所有人脸特征的 csv
        if self.get_face_database():
            while stream.isOpened():
                flag, img_rd = stream.read()
                faces = detector(img_rd, 0)
                kk = cv2.waitKey(1)
                # 按下 q 键退出 / Press 'q' to quit
                if kk == ord('q') or kk == ord('Q'):
                    break
                else:
                    self.record_type_list = []
                    self.record_name_list = []
                    self.record_account_list = []
                    self.record_in_time_list = []
                    self.record_out_time_list = []
                    self.record_strangerPic_list = []
                    sqlstr = 'SELECT * FROM record_easy'
                    cursor.execute(sqlstr)
                    rows = cursor.fetchall()
                    for row in rows:
                        self.record_type_list.append(row[0])
                        self.record_name_list.append(row[1])
                        self.record_account_list.append(row[2])
                        self.record_in_time_list.append(row[3])
                        self.record_out_time_list.append(row[4])
                        self.record_strangerPic_list.append(row[5])

                    # print(self.account_list)
                    # print(self.name_list)
                    # print(self.type_list)
                    # print(self.relation_list)
                    # print(self.line_list)
                    # print(self.record_type_list)
                    # print(self.record_name_list)
                    # print(self.record_account_list)
                    # print(self.record_in_time_list)
                    # print(self.record_out_time_list)
                    # print(self.record_strangerPic_list)
                    # print("==================")
                    self.account_list = []
                    self.name_list = []
                    self.type_list = []
                    self.relation_list = []
                    self.line_list = []
                    sqlstr = 'SELECT * FROM member'
                    cursor.execute(sqlstr)
                    rows = cursor.fetchall()
                    for row in rows:
                        self.account_list.append(row[0])
                        self.name_list.append(row[2])
                        self.type_list.append(row[3])
                        self.relation_list.append(row[4])
                        self.line_list.append(row[7])
                    # print(self.account_list)
                    self.draw_note(img_rd)
                    self.features_camera_list = []
                    self.faces_cnt = 0
                    self.pos_camera_list = []
                    self.name_camera_list = []

                    # 2. 检测到人脸 / when face detected
                    if len(faces)==0:
                        self.student_count = 0
                        self.parent_count = 0
                        self.stranger_count = 0
                        self.teacher_count =0

                    if len(faces) != 0:
                        # 3. 获取当前捕获到的图像的所有人脸的特征，存储到 self.features_camera_list
                        # 3. Get the features captured and save into self.features_camera_list
                        for i in range(len(faces)):
                            shape = predictor(img_rd, faces[i])
                            self.features_camera_list.append(face_reco_model.compute_face_descriptor(img_rd, shape))

                        # 4. 遍历捕获到的图像中所有的人脸 / Traversal all the faces in the database
                        for k in range(len(faces)):
                            # print("##### camera person", k + 1, "#####")
                            # 让人名跟随在矩形框的下方
                            # 确定人名的位置坐标
                            # 先默认所有人不认识，是 unknown
                            # Set the default names of faces with "unknown"
                            self.name_camera_list.append("unknown")

                            # 每个捕获人脸的名字坐标 / Positions of faces captured
                            self.pos_camera_list.append(tuple(
                                [faces[k].left(), int(faces[k].bottom() + (faces[k].bottom() - faces[k].top()) / 4)]))

                            # 5. 对于某张人脸，遍历所有存储的人脸特征
                            # For every faces detected, compare the faces in the database
                            e_distance_list = [0,1]
                            for i in range(len(self.features_known_list)):
                                # 如果 person_X 数据不为空
                                if str(self.features_known_list[i][0]) != '0.0':
                                    # print("with member", self.name_known_list[i], "the e distance: ", end='')
                                    e_distance_tmp = self.return_euclidean_distance(self.features_camera_list[k],
                                                                                    self.features_known_list[i])
                                    # print(e_distance_tmp)
                                    e_distance_list.append(e_distance_tmp)
                                else:
                                    # 空数据 person_X
                                    e_distance_list.append(999999999)
                            # 6. 寻找出最小的欧式距离匹配 / Find the one with minimum e distance
                            similar_person_num = e_distance_list.index(min(e_distance_list))
                            # print("Minimum e distance with person", self.name_known_list[similar_person_num])

                            if min(e_distance_list) < 0.5:
                                self.name_camera_list[k] = self.name_known_list[similar_person_num]
                                # print("May be person " + str(self.name_known_list[similar_person_num]))
                            else:
                                print("Unknown person")

                            # 矩形框 / Draw rectangle
                            for kk, d in enumerate(faces):
                                # 绘制矩形框
                                cv2.rectangle(img_rd, tuple([d.left(), d.top()]), tuple([d.right(), d.bottom()]),
                                              (0, 255, 255), 2)
                            print('\n')

                        self.faces_cnt = len(faces)

                        # 7. 在这里更改显示的人名 / Modify name if needed
                        #self.modify_name_camera_list()
                        # 8. 写名字 / Draw name
                        # self.draw_name(img_rd)
                        img_with_name = self.draw_name(img_rd)
                    else:
                        img_with_name = img_rd

                # print("Faces in camera now:", self.name_camera_list, "\n")
                #把顯示視窗變成全螢幕
                # cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
                # cv2.setWindowProperty("camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

                cv2.imshow("camera", img_with_name)

                # 9. 更新 FPS / Update stream FPS
                self.update_fps()

    # OpenCV 调用摄像头并进行 process
    def run(self):


        cap = cv2.VideoCapture(0)
        cap.set(3, 480)
        self.process(cap)

        cap.release()
        cv2.destroyAllWindows()


def main():
    Face_Recognizer_con = Face_Recognizer()
    Face_Recognizer_con.run()


if __name__ == '__main__':
    main()