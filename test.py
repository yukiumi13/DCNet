from main import *

samplePath = os.listdir('../testset/sample')
labelPath = os.listdir('../testset/label')
data1 = ImageDataset('../testset/sample/' + samplePath[0], '../testset/label/' + labelPath[0])
data = DataLoader(data1, batch_size=4, shuffle=True, pin_memory=False)
test(data)
