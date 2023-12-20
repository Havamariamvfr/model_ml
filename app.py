from flask import Flask, jsonify,request
import tensorflow as tf
import numpy as np
from numpy import genfromtxt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import csv
from collections import defaultdict

app = Flask(__name__)

# function for model

def gen_user_vecs(user_vec, num_items):
    """ given a user vector return:
        user predict maxtrix to match the size of item_vecs """
    user_vecs = np.tile(user_vec, (num_items, 1))
    return user_vecs

def getpredict(y_p, item, animal_dict, maxcount=10):
    """ print results of prediction of a new user. inputs are expected to be in
        sorted order, unscaled. """
    count = 0
    results = []

    for i in range(0, y_p.shape[0]):
        if count == maxcount:
            break
        count += 1
        animal_id = item[i, 0].astype(int)
        results.append(animal_dict[animal_id]['title'])

    return results  

# function for model

# Load Data

item_vecs = genfromtxt('content_item_vecs.csv', delimiter=',')
item_train = genfromtxt('content_item_train.csv', delimiter=',')
user_train = genfromtxt('content_user_train.csv', delimiter=',')
y_train    = genfromtxt('content_y_train.csv', delimiter=',')
animal_dict = defaultdict(dict)
count = 0
with open('content_animal_list.csv', newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='"')
        for line in reader:
            if count == 0:
                count += 1  #skip header
                #print(line) print
            else:
                count += 1
                animal_id = int(line[0])
                animal_dict[animal_id]["title"] = line[1]

# Load Data
model = tf.keras.models.load_model('model.h5',compile=False)

@app.route("/predict",methods=["GET","POST"])
def predict():
    if request.method =="POST":
        try:
            data = request.json
            new_user_id = 2023
            new_perawatan = data.get("perawatan")
            new_ukuran = data.get("ukuran")
            new_aktifitas = data.get("aktifitas")
            new_agresif = data.get("agresif")
            new_kesulitan = data.get("kesulitan")
            new_harga = data.get("harga")
            new_ave = (new_perawatan+new_ukuran+new_aktifitas+new_agresif+new_kesulitan+new_harga)/6
            new_count = new_perawatan+new_ukuran+new_aktifitas+new_agresif+new_kesulitan+new_harga

            user_vec = np.array([[new_user_id, new_count, new_ave,
                                new_perawatan, new_ukuran, new_aktifitas, new_agresif,
                                new_kesulitan, new_harga]])
            
            scalerItem = StandardScaler()
            scalerItem.fit(item_train)

            scalerUser = StandardScaler()
            scalerUser.fit(user_train)

            scalerTarget = MinMaxScaler((-1, 1))
            scalerTarget.fit(y_train.reshape(-1, 1))

            # generate and replicate the user vector to match the number animals in the data set.
            user_vecs = gen_user_vecs(user_vec,len(item_vecs))

            # scale our user and item vectors
            suser_vecs = scalerUser.transform(user_vecs)
            sitem_vecs = scalerItem.transform(item_vecs)

            # make a prediction
            y_p = model.predict([suser_vecs[:, 3:], sitem_vecs[:, 1:]])

            # unscale y prediction
            y_pu = scalerTarget.inverse_transform(y_p)

            # sort the results, highest prediction first
            sorted_index = np.argsort(-y_pu,axis=0).reshape(-1).tolist()  #negate to get largest rating first
            sorted_ypu   = y_pu[sorted_index]
            sorted_items = item_vecs[sorted_index]  #using unscaled vectors for display

            data = getpredict(sorted_ypu, sorted_items, animal_dict, maxcount = 10)
                        
            return jsonify({
                "hewan 1": data[0],
                "hewan 2": data[1],
                "hewan 3": data[2],
                "hewan 4": data[3],
                "hewan 5": data[4],
                "hewan 6": data[5],
                "hewan 7": data[6],
                "hewan 8": data[7],
                "hewan 9": data[8],
                "hewan 10": data[9]
            })
        except Exception as e:
            return jsonify({'error': str(e)})
    else:
        return jsonify({
            "status":{
                "code":405,
                "message":"method not allowed"
            },
            "data":None,
        }),405



if __name__ == "__main__":
    app.run()
