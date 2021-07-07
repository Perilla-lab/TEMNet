import os, sys, csv
from pathlib import Path
from tkinter import *
from tkinter import filedialog as fd
from PIL import Image, ImageTk
import webbrowser
import tensorflow as tf
# sys.path.append(os.path.abspath(os.path.join('..', 'rcnn')))
from config import Config
import predict as P

IMG_FORMAT = ('.tif','.png','.jpg','.jpeg','.bpm','.eps')

DATA =[
    'single',
    'multiple'
]
BACKBONES = [
"temnet",
"resnet101",
"resnet101v2"
] #etc

DEFAULT_FONT = ("shanti", 10, "bold")
BUTTON_FONT = ("Raleway", 12, "bold")

BUTTON_COLOR = "#234042"
FRAME_COLOR = "#67BCC2"
BG_COLOR = "#C1DADB"

def callback(url):
    """
    Opens a url on the browser. It tries the most popular browsers in sequential order or prints an error.
    """
    try:
        webbrowser.get('firefox').open_new_tab(url)
    except webbrowser.Error:
        try:
            webbrowser.get('chrome').open_new_tab(url)
        except webbrowser.Error:
            try:
                webbrowser.get('safari').open_new_tab(url)
            except webbrowser.Error:
                try:
                    webbrowser.get('opera').open_new_tab(url)
                except webbrowser.Error:
                    try:
                        webbrowser.get('chromium').open_new_tab(url)
                    except webbrowser.Error:
                            print(f'No browser found, please install firefox or chrome and open {url}')

def write_counts_csv(csvname, img_names, counts):
  csvname = Path(csvname)
  csvname.parent.mkdir(parents=True, exist_ok=True)
  with open(csvname, mode='w', newline='') as labels:
    fields = ['filename',
              'eccentric',
              'mature',
              'immature']

    writer = csv.DictWriter(labels,
                            fieldnames=fields,
                            dialect='excel',
    quoting=csv.QUOTE_MINIMAL)

    writer.writeheader()
    for i in range(len(counts)):
      writer.writerow({'filename': img_names[i],
                       'eccentric': counts[i][0],
                       'mature': counts[i][1],
                       'immature': counts[i][2]
      })

def save_imgs(images, counts, img_names, backbone):
    """
    Saves images to a directory.
    """
    initialdir = str(Path.home()/'Downloads') if 'win' in sys.platform else None #Windows is stupid with read write directions so files can only be saved to Downloads, target this as the initial folder
    dirname = fd.askdirectory(title="Select where to save your data", initialdir=initialdir)
    write_counts_csv(os.path.normpath(os.path.join(dirname, 'rcnn_preds_'+backbone+'.csv')), img_names, counts)
    for i, img in enumerate(images):
        if img.mode != "RGB": img = img.convert("RGB")
        if i%2:
            save_name = 'rcnn_counts_'+backbone+'_'+str(img_names[i//2])+'.png'
        else:
            save_name = 'rcnn_preds_'+backbone+'_'+str(img_names[i//2])+'.png'
#        with open(os.path.normpath(os.path.join(dirname,save_name), 'w') as write_file:
        try:
            print(save_name)
            img.save(save_name, dpi=(300,300))
        except PermissionError as e:
            print(save_name)
            img.save(dirname+'/'+save_name, dpi=(300,300))            

def display_textbox(content, row, col, window):
    """
    Creates a textbox at a given row and column of the GUI with text content.
    """
    text_box = Text(root, height=1, width=30, padx=10, pady=10)
    text_box.insert(1.0, content)
    # text_box.tag_configure("center", justify="center")
    # text_box.tag_add("center", 1.0, "end")
    text_box.grid(column=col, row=ro, sticky=W, padx=25, pady=25)

def extract_path(textvar):
    """
    Sets a text variable as the path to a file chosen by the user
    """
    filename = fd.askopenfilename(title="Select your weights")
    textvar.set(filename)
    print(textvar.get())

def extract_dir(textvar):
    """
    Sets a text variable as the path to a directory chosen by the user
    """
    dirname = fd.askdirectory(title="Select your data")
    textvar.set(dirname)
    print(textvar.get())


def resize_img(img):
    """
    Resizes a Pillow Image to a width and height fit for displaying in the GUI (usually h=300, w=350 or the closes to this)
    """
    width, height = int(img.size[0]), int(img.size[1])
    if width > height:
        height = int(350/width*height)
        width = 350
    elif height > width:
        width = int(300/height*width)
        height = 300
    else:
        width, height = 300,300
    img = img.resize((width, height))
    return img

def display_image(img, row, column, rowspan=1, columnspan=1, padx=0, pady=0):
    """
    Displays an Pillow image in a specific row and column of the GUI
    """
    img = resize_img(img)
    frame = Frame(main, width=img.size[0]+25, height=img.size[1]+25, bg=FRAME_COLOR)
    frame.grid(columnspan=columnspan, rowspan=rowspan, row=row, column=column)
    img = ImageTk.PhotoImage(img)
    img_label = Label(image=img, bg="white")
    img_label.image = img
    img_label.grid(row=row, column=column, rowspan=rowspan, columnspan=columnspan, padx=padx, pady=pady)
    return img_label, frame

def display_button(url, row, column, sticky=None, funct=None):
    img = Image.open(url)
    #resize image
    img = img.resize((25,25))
    img = ImageTk.PhotoImage(img)
    img_label = Button(image=img, command=funct, width=25, height=25, bg=BUTTON_COLOR)
    img_label.image = img
    img_label.grid(column=column, row=row, sticky=sticky)

def open_images(data_string, textvar):
    """
    Retrieves a single image or a set of images from a path textvar and displays them in the GUI.
    Also updates global image lists and displays the new images.
    """
    global SRC_IMGS
    global SRC_IMG_IX
    global SRC_IMG_DISP
    global SRC_IMG_COUNTER_TEXT
    global SRC_IMG_FRAME
    # Reset source image variables when we open files
    SRC_IMG_IX = 0
    SRC_IMGS = []
    images = []
    if data_string == 'single': #Open one image
        extract_path(textvar)
        images.append(Image.open(textvar.get()))
    elif data_string == 'multiple':
        extract_dir(textvar)
        IMAGES_PATH = textvar.get()
        print(IMAGES_PATH)
        images_ids = next(os.walk(IMAGES_PATH))[2]#All file names in IMAGES_PATH
        for img_name in images_ids:
            if img_name.endswith(IMG_FORMAT):
                try:
                    img = Image.open(os.path.join(IMAGES_PATH, img_name))
                    images.append(img)
                except:
                    print(f"Can't open {os.path.join(IMAGES_PATH,img_name)}... skipping")
        if(len(images)==0):#No images found? Search directory-wise, i.e. /path/07655/07655.png
            dir_ids = next(os.walk(IMAGES_PATH))[1]#All folder names in IMAGES_PATH
            for dir_id in dir_ids:
                file_ids = next(os.walk(os.path.join(IMAGES_PATH,dir_id)))[2]#All files in the directory
                for img_name in file_ids:
                    if img_name.endswith(IMG_FORMAT):
                        try:
                            img = Image.open(os.path.join(IMAGES_PATH, dir_id, img_name))
                            images.append(img)
                        except:
                            print(f"Can't open {os.path.join(IMAGES_PATH, dir_id, img_name)}... skipping")
    print(f"IMGS length: {len(images)}")
    #Images should be read by now
    for img in images:
        if img.mode != "RGB":
            img = img.convert("RGB")
        SRC_IMGS.append(img)
    img = SRC_IMGS[SRC_IMG_IX]
    # img = resize_img(img)
    SRC_IMG_COUNTER_TEXT.set("Source image "+ str(SRC_IMG_IX+1)+ " of "+str(len(SRC_IMGS)))

    if SRC_IMG_FRAME: SRC_IMG_FRAME.grid_forget()
    if SRC_IMG_DISP: SRC_IMG_DISP.grid_forget()
    SRC_IMG_DISP, SRC_IMG_FRAME = display_image(img, row=1 ,column=4,rowspan=3, columnspan=4)

def arrow_src(images, mode='right'):
    """
    Changes the source image to the one next or before in the image array
    """
    global SRC_IMG_IX
    global SRC_IMG_DISP
    global SRC_IMG_FRAME
    global SRC_IMG_COUNTER_TEXT
    if mode=='right':
        SRC_IMG_IX, SRC_IMG_DISP, SRC_IMG_FRAME = right_arrow(images, SRC_IMG_IX, SRC_IMG_DISP, SRC_IMG_FRAME)
    elif mode =='left':
        SRC_IMG_IX, SRC_IMG_DISP, SRC_IMG_FRAME = left_arrow(images, SRC_IMG_IX, SRC_IMG_DISP, SRC_IMG_FRAME)
    SRC_IMG_COUNTER_TEXT.set("Source image "+ str(SRC_IMG_IX + 1) + " of "+str(len(images)))

def arrow_pred(images, mode='right'):
    """
    Changes the predicted image to the one next or before in the image array
    """
    global PRED_IMG_IX
    global PRED_IMG_DISP
    global PRED_IMG_FRAME
    global PRED_IMG_COUNTER_TEXT
    if mode=='right':
        PRED_IMG_IX, PRED_IMG_DISP, PRED_IMG_FRAME = right_arrow(images, PRED_IMG_IX, PRED_IMG_DISP, PRED_IMG_FRAME)
    elif mode =='left':
        PRED_IMG_IX, PRED_IMG_DISP, PRED_IMG_FRAME = left_arrow(images, PRED_IMG_IX, PRED_IMG_DISP, PRED_IMG_FRAME)
    PRED_IMG_COUNTER_TEXT.set("Processed image "+ str(PRED_IMG_IX + 1) + " of "+str(len(images)))

def right_arrow(all_images, img_idx, img_disp, frame):
    """
    Changes the display image to the one next to the inputed index
    """
    img_idx = (img_idx + 1)%len(all_images)
    info = img_disp.grid_info()
    img_disp.grid_forget()
    frame.grid_forget()
    img_disp, frame = display_image(all_images[img_idx], row=info['row'], column=info['column'], rowspan=info['rowspan'], columnspan=info['columnspan'], padx=info['padx'], pady=info['pady'])
    return img_idx, img_disp, frame

def left_arrow(all_images, img_idx, img_disp, frame):
    """
    Changes the display image to the one before to the inputed index
    """
    img_idx = (img_idx - 1)%len(all_images)
    info = img_disp.grid_info()
    img_disp.grid_forget()
    frame.grid_forget()
    img_disp, frame = display_image(all_images[img_idx], row=info['row'], column=info['column'], rowspan=info['rowspan'], columnspan=info['columnspan'])
    return img_idx, img_disp, frame

def predict(data='single', path='./assets/sample.png', backbone='temnet', magnification=3000):
    """
    Loads a model according to the backbone required and predicts on all the images loaded.
    Then displayes the processed images with labeled predictions.
    """
    global PRED_IMGS
    global PRED_IMG_IX
    global PRED_IMG_DISP
    global PRED_IMG_COUNTER_TEXT
    global PRED_IMG_FRAME
    global PRED_IMG_NAMES
    global PRED_IMG_COUNTS
    config = Config(backbone=backbone)
    rcnn_loaded = tf.keras.models.load_model(os.path.join(os.getcwd(),backbone))
    rcnn_loaded.summary()
    #Reset pred image variables when generating a new prediction
    PRED_IMG_IX = 0
    PRED_IMGS = []
    pred_imgs = []
    PRED_IMG_NAMES = []
    PRED_IMG_COUNTS = []
    print("Reading from: ", path)
    base_magnification = config.BASE_MAGNIFICATION
    base_crop_size = config.BASE_CROP_SIZE
    base_crop_step = config.BASE_CROP_STEP
    new_crop_size = int(magnification * (base_crop_size/base_magnification) )
    new_crop_step = int(magnification * (base_crop_step/base_magnification) )
    crop_size = (new_crop_size, new_crop_size)
    crop_step = (new_crop_step, new_crop_step)
    if(data == 'single'):
        print(f"Predicting with crop size: {crop_size} and crop step {crop_step}")
        img_name = path.split('/')[-1].split('.')[0] #Take only the number part, that is '/path/to/img/133433.png' -> '133433'
        _, class_ids, scores, pred_img = P.predict_uncropped_image(path, crop_size, crop_step, config, rcnn_loaded, save_fig=False)
        count_img, class_counts = P.visualize_predictions_count(class_ids, scores, img_name, save_fig=False)
        pred_imgs.extend([pred_img, count_img])
        PRED_IMG_NAMES.append(img_name)
        PRED_IMG_COUNTS.append(class_counts)

    elif(data == 'multiple'):
        #Build image paths
        IMAGES_PATH = path
        print(IMAGES_PATH)
        # Read images from train and validation:
        images_ids = next(os.walk(IMAGES_PATH))[2]#All file names in IMAGES_PATH
        image_paths_train = [os.path.join(IMAGES_PATH, img_name) for img_name in images_ids if img_name.endswith(IMG_FORMAT)]
        if(len(image_paths_train)==0):#No images found? Search directory-wise, i.e. /path/07655/07655.png
            dir_ids = next(os.walk(IMAGES_PATH))[1]#All folder names in IMAGES_PATH
            for dir_id in dir_ids:
                file_ids = next(os.walk(os.path.join(IMAGES_PATH,dir_id)))[2]#All files in the directory
                for img_name in file_ids:
                    if img_name.endswith(IMG_FORMAT):
                        image_paths_train += [os.path.join(IMAGES_PATH, dir_name, img_name) for img_name in images_ids]

        #Predict for every image in the set
        for image_path in image_paths_train:
            # img_name = image_path.split('/')[-1].split('.')[0] #Take only the number part, that is '/path/to/img/133433.png' -> '133433'
            img_name = os.path.basename(image_path).split('.')[0] #Take only the number part, that is '/path/to/img/133433.png' -> '133433'
            # _, _, _, pred_img = P.predict_uncropped_image(image_path, crop_size, crop_step, config, rcnn_loaded, save_fig=False)
            _, class_ids, scores, pred_img = P.predict_uncropped_image(image_path, crop_size, crop_step, config, rcnn_loaded, save_fig=False)
            count_img, class_counts = P.visualize_predictions_count(class_ids, scores, img_name, save_fig=False)
            pred_imgs.extend([pred_img,count_img])
            PRED_IMG_NAMES.append(img_name)
            PRED_IMG_COUNTS.append(class_counts)
    #Prediction arrays should be filled
    #Now make the numpy arrays into PIL Images
    for pred_img in pred_imgs:
        w, h, d = pred_img.shape
        pred_img = Image.frombytes("RGBA", (w,h), pred_img.tobytes())
        pred_img = pred_img.convert("RGB")
        PRED_IMGS.append(pred_img)
    pred_img = PRED_IMGS[PRED_IMG_IX]
    PRED_IMG_COUNTER_TEXT.set("Processed image "+ str(PRED_IMG_IX+1)+ " of "+str(len(PRED_IMGS)))

    if PRED_IMG_FRAME: PRED_IMG_FRAME.grid_forget()
    if PRED_IMG_DISP: PRED_IMG_DISP.grid_forget()
    PRED_IMG_DISP, PRED_IMG_FRAME = display_image(pred_img, row=5 ,column=4,rowspan=3, columnspan=4)

main = Tk()

#GLOBAL GUI VARIABLES
SRC_IMGS = []
SRC_IMG_IX = 0
SRC_IMG_DISP = None
SRC_IMG_COUNTER_TEXT = StringVar()
SRC_IMG_FRAME = None

PRED_IMGS = []
PRED_IMG_IX = 0
PRED_IMG_DISP = None
PRED_IMG_COUNTER_TEXT = StringVar()
PRED_IMG_FRAME = None
PRED_IMG_NAMES = []
PRED_IMG_COUNTS = []

main.title("TEMNet")
main.geometry('+%d+%d'%(350,10)) #place GUI at x=350, y=10
main.configure(bg=BG_COLOR)

########## OPTIONS AREA (LEFT) #########################
#header area - logo & browse button
# header = Frame(main, width=600, height=175, bg="white")
# header.grid(columnspan=4, rowspan=2, row=0)

title_img = Image.open('./assets/TEMNet.png')
w, h = title_img.size
w = int(80/h*w)
h = 80
title_img = title_img.resize((w,h))
title_img = ImageTk.PhotoImage(title_img)
title = Label(main, image=title_img, text="TEMNet", font=("arial", 20), bg=BG_COLOR)
title.grid(columnspan=1, rowspan=2, row=0, column=1, pady=25)

options_lbl = Label(main, text="Options:", font=("shanti",20),bg=BG_COLOR)
options_lbl.grid(column = 0, columnspan=4, row=2, sticky=W, padx=25, pady=10)

backbone_lbl = Label(main, text="Backbone:", font=DEFAULT_FONT,bg=BG_COLOR)
backbone_lbl.grid(column = 0, columnspan=2, row=3, sticky=W, padx=35, pady=10)
backbone = StringVar(main)
backbone.set(BACKBONES[0])
backbone_menu = OptionMenu(main, backbone, *BACKBONES)
backbone_menu.grid(column = 1, columnspan=2, row=3, sticky=W, padx=15, pady=10)

# weights_lbl = Label(main, text="Weights:", font=DEFAULT_FONT,bg=BG_COLOR)
# weights_lbl.grid(column = 0, columnspan=1, row=4, sticky=W, padx=35, pady=10)
# weights_path = StringVar(main)
# weights_path.set('')
# weights_textbox = Entry(main, textvariable=weights_path)
# weights_textbox.grid(column = 1, columnspan=1, row=4, sticky=W, padx=15, pady=10)
# display_button('./assets/browse.png',row=4,column=3, sticky=W, funct=lambda:extract_path(weights_path))


data_lbl = Label(main, text="Images:", font=DEFAULT_FONT,bg=BG_COLOR)
data_lbl.grid(column = 0, columnspan=1, row=4, sticky=W, padx=35, pady=10)
data_path = StringVar(main)
data_path.set('')
data_textbox = Entry(main, textvariable=data_path)
data_textbox.grid(column = 1, columnspan=1, row=4, sticky=W, padx=15, pady=10)
data_string = StringVar(main)
data_string.set(DATA[0])
data_menu = OptionMenu(main, data_string, *DATA)
data_menu.grid(column = 2, columnspan=1, row=4, sticky=W, padx=15, pady=10)
# display_button('./assets/browse.png',row=5,column=3, sticky=W, funct=lambda: extract_dir(data_path) if data_string.get()=='multiple' else extract_path(data_path))
display_button('./assets/browse.png',row=4,column=3, sticky=W,
               funct=lambda: open_images(data_string.get(), data_path))


magnification_lbl = Label(main, text="Magnification:", font=DEFAULT_FONT,bg=BG_COLOR)
magnification_lbl.grid(column = 0, columnspan=1, row=5, sticky=W, padx=35, pady=10)
magnification_val = StringVar(main)
magnification_val.set('30000')
magnification_textbox = Entry(main, textvariable=magnification_val)
magnification_textbox.grid(column = 1, columnspan=1, row=5, sticky=W, padx=15, pady=10)

predict_btn = Button(main, text="PREDICT!",
                  command=lambda: predict(data_string.get(), data_path.get(), backbone.get(), int(magnification_val.get())),
                  font=("Raleway",16, 'bold'), bg=BUTTON_COLOR, fg="white", height=2, width=25)
predict_btn.grid(column=0, row=6, columnspan=2, padx=10, pady=00)

save_btn = Button(main, text="SAVE IMAGES!",
                  command=lambda: save_imgs(PRED_IMGS, PRED_IMG_COUNTS, PRED_IMG_NAMES, backbone.get()),
                  font=BUTTON_FONT, bg=BUTTON_COLOR, fg="white", height=1, width=15)
save_btn.grid(column=2, row=6, columnspan=2, padx=10, pady=00)


copyright_lbl = Label(main, text="Juan Rey, Alex Bryer, Hagan Beatson, Christian Lantz, Juan Perilla \n Perilla Labs (c) University of Delaware 2021, Apache-2.0 License", font=('shanti', 10, 'italic'),bg=BG_COLOR)
copyright_lbl.grid(column = 0, columnspan=3, row=7, sticky=E, padx=15, pady=10)

#Acknowledgements
ud_logo = Image.open('./assets/Logos.png')
w, h = ud_logo.size
w = int(65/h*w)
h = 65
ud_logo = ud_logo.resize((w,h))
ud_logo = ImageTk.PhotoImage(ud_logo)
ud_lbl = Label(main, image=ud_logo, text="UD", font=("arial", 20), bg=BG_COLOR)
ud_lbl.grid(columnspan=1, rowspan=2, row=0, column=0, pady=15)


repo_lbl = Label(main, text="Source Code!", font=("shanti", 10,'bold'), bg=BG_COLOR)
repo_lbl.grid(columnspan=1, rowspan=1, row=0, column=2, sticky=S, pady=0, padx=15)
github_logo = Image.open('./assets/github-512.webp')
w, h = github_logo.size
w = int(35/h*w)
h = 35
github_logo = github_logo.resize((w,h))
github_logo = ImageTk.PhotoImage(github_logo)
github_lbl = Label(main, image=github_logo, text="GITHUB", font=("arial", 20), bg=BG_COLOR)
github_lbl.bind("<Button-1>", lambda e: callback("https://github.com/Perilla-lab/TEMNet"))
github_lbl.grid(columnspan=1, rowspan=1, row=1, column=2, sticky=N, pady=0)

########## IMAGES AREA (RIGHT) ##########################

# img_menu = Frame(main, width=700, height=60, bg="white")
# img_menu.grid(columnspan=4, rowspan=10, row=0, column=4)

# Original images
#Sample img
source_img = Image.open('./assets/sample.png')
SRC_IMGS.append(source_img)
SRC_IMG_COUNTER_TEXT.set("Source image "+ str(SRC_IMG_IX+1) + " of "+str(len(SRC_IMGS)))

ogimg_lbl = Label(main, textvariable=SRC_IMG_COUNTER_TEXT, font=('shanti',10), bg=BG_COLOR, width=25)
ogimg_lbl.grid(column = 5, columnspan=2, row=0, padx=15, pady=15)
#Left arrow
display_button('./assets/arrow_l.png', column=4, row=0, sticky=E, funct=lambda: arrow_src(SRC_IMGS, 'left'))
#right arrow
display_button('./assets/arrow_r.png', column=7, row=0, sticky=W, funct=lambda: arrow_src(SRC_IMGS, 'right'))

SRC_IMG_DISP, SRC_IMG_FRAME = display_image(source_img, row=1, column=4, rowspan=3, columnspan=4, padx=35)

# Predicted images
#Sample img
pred_img = Image.open('./assets/sample_preds.png')
PRED_IMGS.append(pred_img)
PRED_IMG_COUNTER_TEXT.set("Processed image "+ str(PRED_IMG_IX+1) + " of "+str(len(PRED_IMGS)))

predimg_lbl = Label(main, textvariable=PRED_IMG_COUNTER_TEXT, font=('shanti',10),bg=BG_COLOR, width=25)
predimg_lbl.grid(column = 5, columnspan=2, row=4, padx=15, pady=15)
#Left arrow
display_button('./assets/arrow_l.png', column=4, row=4, sticky=E, funct=lambda: arrow_pred(PRED_IMGS,'left'))
#right arrow
display_button('./assets/arrow_r.png', column=7, row=4, sticky=W, funct=lambda: arrow_pred(PRED_IMGS,'right'))

PRED_IMG_DISP, PRED_IMG_FRAME = display_image(pred_img, row=5, column=4, rowspan=3, columnspan=4, padx=35)

# main.wm_attributes('-transparentcolor', 'grey')
main.mainloop()
