try:
  import unzip_requirements
except ImportError:
  pass


import json
import boto3
import numpy as np
import base64
import io
import cv2



#--- pixel values of airfield
arifield_pix_value = []
for i in range(184, 189):
    for j in range(184, 189):
        for k in range(203, 206):
            pix_val = (i, j, k)
            arifield_pix_value.append(pix_val)


def decode_base64_to_cv2(img_b64):
    img = base64.urlsafe_b64decode(img_b64)
    img_io = io.BytesIO(img)
    img_np = np.frombuffer(img_io.read(), dtype=np.uint8)
    img_cv2 = cv2.imdecode(img_np, cv2.IMREAD_COLOR)
    pic = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    return pic


#--- return 256x256 number of (r,g,b) tuples
def pic_val_count(pic):
    reshaped_pic = np.reshape(pic, (pic.shape[0] * pic.shape[1], 3))
    reshaped_pic = reshaped_pic.tolist()
    reshaped_pic = [tuple(pixel) for pixel in reshaped_pic]

    col_count = []
    for i in set(reshaped_pic):
        (col_val, num_pic) = i, reshaped_pic.count(i)
        col_count.append((col_val, num_pic))
    return col_count

#--- return if the image is airfield
def classify_airfield_image(img_cv2):
    result = 0
    for pic_val, num in pic_val_count(img_cv2):
        if (pic_val in arifield_pix_value) & (num > 30):
            result = 1
    return result


def inferHandler(event, context):
    body_txt = event['body']
    body_json = json.loads(body_txt)
    z = body_json['z'] 
    x = body_json['x']
    y = body_json['y']
    tile_base64 = body_json['tile_base64']
    img_cv2 = decode_base64_to_cv2(tile_base64)

    predictions = classify_airfield_image(img_cv2)

#    AWS_BUCKET_NAME = 'wmts.maprover.io'
#    AWS_BUCKET_NAME = 'md-maprover'
    if predictions == 0:
        dic = False
    else:
        dic = True

    # s3 = boto3.resource('s3')
    # bucket = s3.Bucket(AWS_BUCKET_NAME)
    # path = z + '/' + x + '/' + y
    # data = base64.b64decode(tile_base64)
    

    # bucket.put_object(
    #     ContentType='image/png',
    #     Key=path,
    #     Body=data,
    #     ACL='public-read'
    # )


    response = {
        "statusCode": 200,
        "headers": {'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
        "body": json.dumps({'RailClass': dic})
    }
    
    return response
