train_paths = [
    '/mnt/disk2/cyl/library/dataset/DataDayCount/Daycount2014.csv',
    '/mnt/disk2/cyl/library/dataset/DataDayCount/Daycount2015.csv',
    '/mnt/disk2/cyl/library/dataset/DataDayCount/Daycount2016.csv' ,
    ]

test_paths = ['/mnt/disk2/cyl/library/dataset/DataDayCount/Daycount2017.csv' ,]   #/DataHourCount/Hourcount2017.csv/DataDayCount/Daycount2017.csv

datatype = 'DayCount'   #DayCount/Hourcount/Min30count
model = 'LCTC'
input_size = 1  #输入特征（借书量）
input_window = 28   #输入序列长度
output_window = 1   #输出序列长度
step = 1            #读取数据的步长
batch_size = 16

hidden_size = 64
num_layers = 4
num_heads = 4

save_path = './MyWeight/' + str(datatype)+'_'+ str(model)+'_in'+str(input_window) + '_out'+ str(output_window) + '_step' + str(step) + '_hs'+str(hidden_size) + '_nl'+str(num_layers) + '_nh'+ str(num_heads) + '/'
test_weights_path = save_path + 'best.pth'
