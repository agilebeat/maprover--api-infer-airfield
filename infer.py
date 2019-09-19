try:
  import unzip_requirements
except ImportError:
  pass


import json
import boto3
from PIL import Image
from sklearn.svm import SVC
from joblib import dump, load
import numpy as np
import base64
import io

clf_fromfile = load('sklearn-models/airfield_svm.joblib')
    
    
def run_classify_image(img):
    
    f = gfile.FastGFile("tf-models/tf_model.pb", 'rb')
    graph_def = tf.GraphDef()
   # Parses a serialized binary message into the current message.
    graph_def.ParseFromString(f.read())
    f.close()

    sess = tf.Graph()
    with sess.as_default() as graph:
        tf.import_graph_def(graph_def)
        softmax_tensor = sess.get_tensor_by_name('import/activation_15_2/Softmax:0')

    with tf.Session(graph=graph) as sess:
        predictions = sess.run(softmax_tensor, {'import/conv2d_6_input_2:0': img})
         
    return predictions

def evaluate_image(image):
    image_pil = image
    image_np = np.array(image_pil)[...,:3]
    image_flat_vector = image_np.flatten()
    return clf_fromfile.predict([image_flat_vector])[0]
        


def inferHandler(event, context):
    body_txt = event['body']
    body_json = json.loads(body_txt)
    z = body_json['z'] 
    x = body_json['x']
    y = body_json['y']
    tile_base64 = body_json['tile_base64'] 
    

    img = base64.urlsafe_b64decode(tile_base64)
    img = io.BytesIO(img)
    img = Image.open(img)


    predictions = evaluate_image(img)

#    AWS_BUCKET_NAME = 'wmts.maprover.io'
    AWS_BUCKET_NAME = 'md-maprover'
    if predictions == 0:
        dic = False
    else:
        dic = True

    s3 = boto3.resource('s3')
    bucket = s3.Bucket(AWS_BUCKET_NAME)
    path = z + '/' + x + '/' + y
    data = base64.b64decode(tile_base64)
    

    bucket.put_object(
        ContentType='image/png',
        Key=path,
        Body=data,
        ACL='public-read'
    )


    response = {
        "statusCode": 200,
        "headers": {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        "body": json.dumps({'RailClass': dic})
    }
    
    return response
