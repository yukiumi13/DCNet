import numpy as np
'''
for file in os.listdir('../SrcData/Sample'):
    try:
        a = nib.load('../SrcData/Sample/' + file).get_fdata()
        print(file)
        print(a.shape)
    except nib.filebasedimages.ImageFileError :
        pass
'''
'''
imgs = []
img = cv2.imread('../CVdataset/HKU-IS/img/1804.png')
print(img.shape)
imgs = imgs.append(img)
print(imgs.shape)
'''
'''
def data_process_img(path, content):

    imgs=[] 
    for i in content :
        img_path = path + '/' + i
        print('processing image', img_path)
        img = cv2.imread(img_path) # cv2.imread是按照BGR的顺序读的图像
        b, g, r = cv2.split(img)
        img = cv2.merge([r, g, b])

        # img[1:20, 1:77, :] = 30
        # img[1:22, 279:384, :] = 30
        # img[360:384, 1:93, :] = 30
        # img[361:384, 329:384, :] = 30

        img = cv2.resize(img, (256, 256))
        # img = img_normalize(img) # 图像标准化
        # io.imshow(img)
        # io.show()
        imgs.append(img)
    
    img = np.stack(imgs, axis=0)
    print(str(len(content)) +' images loaded')
    imgs = np.transpose(imgs,(0,3,1,2))
    print('img shape is ' + str(imgs.shape))

    return imgs

imgs = data_process_img('../CVdataset/HKU-IS/img',['0004.png','0005.png'])
'''
'''
c=[]
a = cv2.imread('../CVdataset/HKU-IS/img/0004.png')
a = cv2.resize(a,(256,256))
c.append(a)
print(a.shape)
b = cv2.imread('../CVdataset/HKU-IS/img/0005.png')
b = cv2.resize(b,(256,256))
c.append(b)
print(b.shape)
c = np.stack(c)
print(c.shape)

dataset = 'NPC-seg'
sample = np.load('../npys/'+dataset+'/IMG/sample/sample.npy')
label = np.load('../npys/'+dataset+'/IMG/label/label.npy')
print(np.mean(label))
'''
import cv2
a = np.random.randn(256,256)
b = cv2.threshold(a,0.5,1,cv2.THRESH_BINARY)
