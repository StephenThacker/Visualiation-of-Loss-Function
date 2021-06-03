custom_objects = {'mean_iou': metrics.mean_iou,
                  'dice_coefficient': metrics.dice_coefficient}

def readimagesintonumpyArray(filepath):
    B = []
    C = []
    for filename in os.listdir(filepath):
        A = filename.split(".")
        if IsInt(A[0]) == True and A[1] == "png":
            path = filepath + filename
            img = io.imread(path, as_gray=True)
            img = np.asarray(img)
            B += [img]
            C += [filename]
    B = np.asarray(B)
    return [B,C]



def makegroundtruth(Ground_truth):
    A= []
    for elem in Ground_truth:
        New = elem
        for i in range(0,len(elem)-1):
            for j in range(0,len(elem)-1):
                New[i][j] = 1 - elem[i][j]
        B = []
        B += [New]
        B += [elem]
        B = np.swapaxes(B,0,1)
        B = np.swapaxes(B,1,2)
        A += [B]
    A = np.asarray(A)
    return A

def makegroundtruth2(Ground_truth):
    A= []
    for elem in Ground_truth:
        Tempp = np.full((len(elem),len(elem)),1,np.float64)
        New = np.subtract(Tempp,elem)
        B = []
        B += [New]
        B += [elem]
        B = np.swapaxes(B,0,1)
        B = np.swapaxes(B,1,2)
        A += [B]
    A = np.asarray(A)

    return A

def maketrainingdata(train_data):
    B = []
    for elem in train_data:
        new = [elem]
        new = np.swapaxes(new,0,1)
        new = np.swapaxes(new,1,2)
        B += [new]
    B = np.asarray(B)
    return B
  
  def takecenter(groundtruths):
    Array = []
    for img in groundtruths:
        img = np.swapaxes(img,2,1)
        img = np.swapaxes(img,0,1)
        temp1 = img[0][22:110,22:110]
        temp2 = img[1][22:110,22:110]
        array  = np.stack((temp1,temp2),0)
        array = np.swapaxes(array,0,1)
        array = np.swapaxes(array,1,2)
        Array += [array]

    Array = np.asarray(Array)
    return Array
