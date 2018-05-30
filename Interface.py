import tkinter as tk
from tkinter.filedialog import *
from tkinter import ttk
import train_use_kears as train
import cv2
from PIL import Image, ImageTk
import threading
import time
from tkinter import messagebox as mBox
import DetectPossibleknife

class Surface(ttk.Frame):
    pic_path = ""
    viewhigh = 600
    viewwide = 600
    update_time = 0
    thread = None
    thread_run = False
    camera = None

    def __init__(self, win):
        ttk.Frame.__init__(self, win)
        frame_left = ttk.Frame(self)
        frame_right = ttk.Frame(self)
        frame_top = ttk.Frame(self)
        win.title("危险物品识别")
        win.state("zoomed")
        self.pack(fill=tk.BOTH, expand=tk.YES, padx="5", pady="5")
        frame_top.pack(side=TOP)
        frame_left.pack(side=LEFT)
        frame_right.pack(side=RIGHT)

        ttk.Label(frame_left, text='原图：').pack(anchor="n")
        self.image_ct1= ttk.Label(frame_left)
        self.image_ct1.pack(anchor="n")

        ttk.Label(frame_right, text='可疑位置：').pack(anchor="n")
        self.image_ct2 = ttk.Label(frame_right)
        self.image_ct2.pack(anchor="n")

        #定义四个按钮
        model_select=ttk.Button(frame_top, text="选择模型", width=30, command=self.model_select)
        from_pic_ctl = ttk.Button(frame_top, text="来自图片", width=30, command=self.from_pic)
        from_vedio_ctl = ttk.Button(frame_top, text="来自摄像头", width=30, command=self.from_vedio)
        train_model = ttk.Button(frame_top, text="训练模型", width=30, command=self.train_model)
        model_select.grid(row=1,column=0)
        from_vedio_ctl.grid(row=0, column=1)
        from_pic_ctl.grid(row=2, column=1)
        train_model.grid(row=1, column=2)


    def get_imgtk(self, img_bgr):
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        im = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(image=im)
        wide = imgtk.width()
        high = imgtk.height()
        if wide > self.viewwide or high > self.viewhigh:
            wide_factor = self.viewwide / wide
            high_factor = self.viewhigh / high
            factor = min(wide_factor, high_factor)
            wide = int(wide * factor)
            if wide <= 0: wide = 1
            high = int(high * factor)
            if high <= 0: high = 1
            im = im.resize((wide, high), Image.ANTIALIAS)
            imgtk = ImageTk.PhotoImage(image=im)
        return imgtk

    def from_vedio(self):
        if self.thread_run:
            return
        if self.camera is None:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                mBox.showwarning('警告', '摄像头打开失败！');   #弹出消息提示框
                self.camera = None
                return
        self.thread = threading.Thread(target=self.vedio_thread, args=(self,))
        self.thread.setDaemon(True)
        self.thread.start()
        self.thread_run = True

    def from_pic(self):
        self.thread_run = False
        pic_path = askopenfilename(title="选择识别图片", filetypes=[("jpg图片", "*.jpg")])
        if pic_path:
            img_bgr = cv2.imread(pic_path)
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ct1.configure(image=self.imgtk)
            self.judge(img_bgr)

    def model_select(self):
        self.thread_run=False
        model_path = askopenfilename(title="选择模型", filetypes=[("模型文件", "*.h5")])
        if model_path:
            mBox.showinfo('模型加载')
            self.model = train.Model()
            self.model.load_model(model_path)
            img=cv2.imread('./picture/1.jpg')
            self.model.predict(img)
            mBox.showinfo('模型加载完成')

    def train_model(self):
        self.thread_run = True
        self.file_path = askdirectory(title="选择建立模型数据源")
        mBox.showinfo('开始训练')
        dataset = train.Dataset(str(self.file_path))
        dataset.load()
        model = train.Model()
        model.build_model(dataset)
        model.train(dataset)
        model.save_model(self.file_path+'/model')
        mBox.showinfo('训练完成')
        self.thread_run=False

    def judge(self,img):
        imgscene, list = DetectPossibleknife.detectknifeInScene(img)
        h,w=imgscene.shape[0:2]
        print('图片尺寸',h,w)
        # cv2.waitKey(0)
        # 框住目标检测物的矩形边框颜色
        color = (0, 255, 0)
        # 循环检测识别
        for scene in list:
            # 截取可疑部位图像提交给模型识别
            print('list',scene.intBoundingRectX - 10, scene.intBoundingRectY - 10,
                  scene.intBoundingRectWidth +
                   scene.intBoundingRectX + 10,
                   scene.intBoundingRectY + scene.intBoundingRectHeight+ 10)
            y1=scene.intBoundingRectY - 10
            if y1<0:
                y1=0
            x1=scene.intBoundingRectX - 10
            if x1<0:
                x1=0
            y2=scene.intBoundingRectY + scene.intBoundingRectHeight + 10
            if y2>h:
                y2=h
            x2=scene.intBoundingRectX + scene.intBoundingRectWidth + 10
            if x2>w:
                x2=w
            print('x1,y1,x2,y2',x1,y1,x2,y2)
            newimage = imgscene[y1:y2,x1:x2]# 先用y确定高，再用x确定宽
            ID = self.model.predict(newimage)
            # 如果是“危险”
            if ID == 0:

                cv2.rectangle(imgscene, (x1,y1),(x2,y2), color, thickness=2)
                # 文字提示
                cv2.putText(imgscene, 'Warning',
                            (x1+ 50, y1 + 50),  # 坐标
                            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                            1,  # 字号
                            (255, 0, 255),  # 颜色
                            2)  # 字的线宽

            else:
                pass
        self.imgtk2 = self.get_imgtk(imgscene)
        self.image_ct2.configure(image=self.imgtk2)

        cv2.destroyAllWindows()
    @staticmethod
    def vedio_thread(self):
        self.thread_run = True
        predict_time = time.time()
        while self.thread_run:
            _, img_bgr = self.camera.read()
            self.imgtk = self.get_imgtk(img_bgr)
            self.image_ct1.configure(image=self.imgtk)
            #设置时间间隔，间隔两秒检测一次
            if time.time() - predict_time > 2:
                self.judge(img_bgr)
                predict_time = time.time()
            # 如果输入q则退出循环
            if 0xFF == ord('q'):
                break
        print("run end")


def close_window():
    print("destroy")
    if surface.thread_run:
        surface.thread_run = False
        surface.thread.join(2.0)
    win.destroy()


if __name__ == '__main__':
    win = tk.Tk()
    surface = Surface(win)
    win.protocol('WM_DELETE_WINDOW', close_window)
    win.mainloop()


