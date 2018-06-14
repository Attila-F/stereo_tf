import numpy as np
import tensorflow as tf
import os
from tqdm import tqdm
from scipy import ndimage
from PIL import Image
import argparse
import matplotlib.pyplot as plt

import config as cfg

parser = argparse.ArgumentParser(description='Select mode')
parser.add_argument('--mode', dest='mode', help='preprocess, train or test')
args = parser.parse_args()

#############################################
################## Utils ####################
#############################################

def disparity_to_color(I):
    
    _map = np.array([[0,0, 0, 114], [0, 0, 1, 185], [1, 0, 0, 114], [1, 0, 1, 174], 
                    [0, 1, 0, 114], [0, 1, 1, 185], [1, 1, 0, 114], [1, 1, 1, 0]]
                   )      
    max_disp = 1.0*I.max()
    I = np.minimum(I/max_disp, np.ones_like(I))
    
    A = I.transpose()
    num_A = A.shape[0]*A.shape[1]
    
    bins = _map[0:_map.shape[0]-1,3]    
    cbins = np.cumsum(bins)    
    cbins_end = cbins[-1]
    bins = bins/(1.0*cbins_end)
    cbins = cbins[0:len(cbins)-1]/(1.0*cbins_end)
    
    A = A.reshape(1,num_A)            
    B = np.tile(A,(6,1))        
    C = np.tile(np.array(cbins).reshape(-1,1),(1,num_A))
       
    ind = np.minimum(sum(B > C),6)
    bins = 1/bins
    cbins = np.insert(cbins, 0,0)
    
    A = np.multiply(A-cbins[ind], bins[ind])   
    K1 = np.multiply(_map[ind,0:3], np.tile(1-A, (3,1)).T)
    K2 = np.multiply(_map[ind+1,0:3], np.tile(A, (3,1)).T)
    K3 = np.minimum(np.maximum(K1+K2,0),1)
    return np.reshape(K3, (I.shape[1],I.shape[0],3)).T

def get_pixel_error(threshold, pred, gt):
    gt_map = np.zeros(gt[4:-4, 4:-4].shape)
    gt_map[gt[4:-4, 4:-4] > 0] = 1
    gt_sum = np.sum(gt_map)

    pixel_error = np.abs(pred-gt[4:-4, 4:-4])
    pixel_error[gt[4:-4, 4:-4] == 0] = 0
    pixel_error[pixel_error <= threshold] = 0
    pixel_error[pixel_error > threshold] = 1
    return np.sum(pixel_error)/gt_sum

#############################################
############## Preprocessing ################
#############################################

def load_disparity_image(path):
    return ndimage.imread(path,flatten=True)/256.0

def normalize_image(image):
    image = np.array(image, dtype=np.float32)
    return (image-np.mean(image)) / np.std(image)
    
def load_images(image_2_paths, image_3_paths, disparity_paths):
    list_2=[]
    list_3=[]
    list_d=[]
    for i in tqdm(range(len(image_2_paths)), desc="Preprocessing inputs"):
        list_2.append(normalize_image(ndimage.imread(image_2_paths[i])))
        list_3.append(normalize_image(ndimage.imread(image_3_paths[i])))
        list_d.append(load_disparity_image(disparity_paths[i]))
    return list_2, list_3, list_d

def split_data(path, validation_split):
    order = list(range(0, 200))
    np.random.shuffle(order)
    val_index = order[0:validation_split]
    list_2=[]
    list_3=[]
    list_d=[]
    list_2_val=[]
    list_3_val=[]
    list_d_val=[]

    for i in range(0,200):
        if i in val_index:
            list_2_val.append(os.path.join(path, "training/image_2",    str(i).zfill(6) + "_10.png"))
            list_3_val.append(os.path.join(path, "training/image_3",    str(i).zfill(6) + "_10.png"))
            list_d_val.append(os.path.join(path, "training/disp_noc_0", str(i).zfill(6) + "_10.png"))
 
        else:
            list_2.append(os.path.join(path, "training/image_2",    str(i).zfill(6) + "_10.png"))
            list_3.append(os.path.join(path, "training/image_3",    str(i).zfill(6) + "_10.png"))
            list_d.append(os.path.join(path, "training/disp_noc_0", str(i).zfill(6) + "_10.png"))

    images_2, images_3, images_d = load_images(list_2, list_3, list_d)
    images_2_val, images_3_val, images_d_val = load_images(list_2_val, list_3_val, list_d_val)
    
    np.save('data/images_2', images_2)
    np.save('data/images_3', images_3)
    np.save('data/images_d', images_d)
    np.save('data/images_2_val', images_2_val)
    np.save('data/images_3_val', images_3_val)
    np.save('data/images_d_val', images_d_val)

    return  images_d, images_d_val

def get_valid_locations(list_d, receptive_field, max_disparity):
    halfrecp = int(receptive_field/2)
    valid_locations = []
    for i in tqdm(range(len(list_d)), desc="Extracting valid pixels"):
        filt = np.where((list_d[i] > 2) & (list_d[i] <= max_disparity - 2))
        filt = list(map(list, zip(*filt)))
        filt = [x_y_pair for x_y_pair in filt if x_y_pair[0] > halfrecp]
        filt = [x_y_pair for x_y_pair in filt if x_y_pair[0] < list_d[i].shape[0] - halfrecp]
        #filt = [x_y_pair for x_y_pair in filt if x_y_pair[1] > halfrecp + max_disparity + list_d[i][x_y_pair[0]][x_y_pair[1]]]
        filt = [x_y_pair for x_y_pair in filt if x_y_pair[1] > halfrecp + max_disparity]
        filt = [x_y_pair for x_y_pair in filt if x_y_pair[1] < list_d[i].shape[1] - halfrecp]
        valid_locations.append(filt)
    return valid_locations

def preprocess(): 
    images_d, images_d_val = split_data(cfg.KITTIPATH, 40)

    locations = get_valid_locations(images_d,receptive_field,cfg.MAX_DISPARITY)
    locations_val = get_valid_locations(images_d_val,receptive_field,cfg.MAX_DISPARITY)
    np.save('data/locations', locations)
    np.save('data/locations_val', locations_val)

#############################################
############### Data Loader #################
#############################################

def load_preprocessed_data():
    sources = ['images_2', 'images_3', 'images_d', 'images_2_val', 'images_3_val', 'images_d_val', 'locations', 'locations_val']
    dest = []
    for source in tqdm(sources, desc='Loading preprocessed data'):
        dest.append(np.load(os.path.join(os.getcwd(), 'data', source+'.npy')))

    return dest

def get_random_patch(list_2, list_3, list_label, list_valid_pixels, receptive_field_size, max_disparity, batch_size, loss_weights):
    halfrecp=int(receptive_field_size/2)

    batch_2 = np.zeros((0, receptive_field_size, receptive_field_size, 3), dtype=np.float32)
    batch_3 = np.zeros((0, receptive_field_size, receptive_field_size + max_disparity, 3), dtype=np.float32)
    batch_label = np.zeros((0, max_disparity + 1), dtype=np.float32)

    for batch in range(0,batch_size):
        # Select a random valid patch
        i_img = np.random.randint(len(list_2))
        i_pix = np.random.randint(len(list_valid_pixels[i_img]))

        x = list_valid_pixels[i_img][i_pix][0]
        y = list_valid_pixels[i_img][i_pix][1]

        patch_2 = list_2[i_img][x - halfrecp : x + halfrecp + 1,
                                y - halfrecp : y + halfrecp + 1]
        patch_3 = list_3[i_img][x - halfrecp : x + halfrecp + 1,
                                y - halfrecp - max_disparity : y + halfrecp + 1]
        label_v = np.zeros([1, max_disparity + 1])

        local_disparity_right_left = int(max_disparity - list_label[i_img][x, y])
        label_v[:, local_disparity_right_left-2:local_disparity_right_left+3] = np.array(loss_weights)
        #[0, 0, 0, 1/20, 4/20, 10/20, 4/20, 1/20, 0, 0, 0, ... 0]

        batch_2 = np.concatenate([batch_2, np.expand_dims(patch_2, axis=0)], axis=0)
        
        batch_3 = np.concatenate([batch_3, np.expand_dims(patch_3, axis=0)], axis=0)
        batch_label = np.concatenate([batch_label, label_v], axis=0)

    return batch_label, batch_2, batch_3

def get_random_images(list_2, list_3, list_label, receptive_field_size):
    halfrecp=int(receptive_field_size/2)

    i_img = np.random.randint(len(list_2))

    image_2 = list_2[i_img]
    image_3 = list_3[i_img]
    image_d = list_label[i_img]

    return np.expand_dims(image_d, 0), np.expand_dims(image_2,0), np.expand_dims(image_3,0)

#############################################
################# Layers ####################
#############################################

def conv(patch, kernel, bias, is_training, reuse, scope):
    with tf.variable_scope(scope,reuse=reuse):
        weights    = tf.get_variable("weights", kernel, initializer = tf.contrib.layers.xavier_initializer_conv2d()) 
        biases     = tf.get_variable("biases", bias, initializer=tf.contrib.layers.xavier_initializer())
        conv       = tf.nn.conv2d(patch, weights, strides=[1, 1, 1, 1], padding='VALID')
        batch_norm = tf.contrib.layers.batch_norm(tf.nn.bias_add(conv,biases), 
                                          center=True, scale=True, 
                                          is_training=is_training, decay=0.9,
                                          scope='bn')
        return batch_norm

def BaseNet(patch, reuse, is_training, channels, filters):
    layer_1 = tf.nn.relu(conv(patch, [3, 3, channels, filters], [filters], is_training, reuse, 'layer_1'))
    layer_2 = tf.nn.relu(conv(layer_1, [3, 3, filters, filters], [filters], is_training, reuse, 'layer_2'))
    layer_3 = tf.nn.relu(conv(layer_2, [3, 3, filters, filters], [filters], is_training, reuse,'layer_3'))
    layer_4 = conv(layer_3, [3, 3, filters, filters], [filters], is_training, reuse, 'layer_4')
    return layer_4
    
def build_graph(patch_2, patch_3, gt_v, channels, filters, max_disparity, learning_rate, global_step, is_training):
    with tf.variable_scope('Net') as scope:
        out_2 = BaseNet(patch=patch_2,reuse=False, is_training=is_training, channels=channels, filters=filters)
        out_3 = BaseNet(patch=patch_3,reuse=True, is_training=is_training, channels=channels, filters=filters)
    
    #(?, 1, 1, 64) (?, 1, 129, 64)
    inner_product = tf.reduce_sum(tf.multiply(out_2, out_3), axis=-1)


    #inner_product = tf.matmul(out_2, tf.transpose(out_3, perm=[0,1,3,2]))
    #(?, 1, 1, 129)
    
    softmax_scores = -tf.nn.log_softmax(inner_product, axis=-1)

    with tf.name_scope('Loss'):
        loss = tf.reduce_mean(tf.reduce_sum(tf.multiply(softmax_scores, gt_v), axis=-1)) # (?,129)*(?,129)-> (?,1)

    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_step = tf.train.AdagradOptimizer(learning_rate).minimize(loss,global_step=global_step)
    return train_step, loss, softmax_scores, out_2, out_3

def init_placeholders(receptive_field_size_h, receptive_field_size_w, channels, max_disparity):    
    with tf.name_scope('image_2'):
        ph_image_2 = tf.placeholder(tf.float32, shape=[None, receptive_field_size_h, receptive_field_size_w, channels])
    
    with tf.name_scope('image_3'):    
        ph_image_3 = tf.placeholder(tf.float32, shape=[None, receptive_field_size_h, receptive_field_size_w + max_disparity, channels])

    with tf.name_scope('label'):
        ph_label = tf.placeholder(tf.float32, shape=[None, max_disparity + 1])

    ph_is_training = tf.placeholder(tf.bool)

    return ph_image_2, ph_image_3, ph_label, ph_is_training

#############################################
################## Train ####################
#############################################

def train_model():
    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = 0.01

    # Decrease learning rate as in the paper
    boundaries = [24000, 32000]
    rates = [0.01, 0.01/5, 0.01/25]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, rates)
    # Load data  
    images_2, images_3, noc_1, images_2_val, images_3_val, noc_1_val, valid_locations, valid_locations_val = load_preprocessed_data()
    # Initialize placeholders
    ph_patch_2, ph_patch_3, ph_label, ph_is_training = init_placeholders(receptive_field, receptive_field, cfg.CHANNELS, cfg.MAX_DISPARITY)
    train_step, cross_entropy, output_layer_t, _, _ = build_graph(patch_2=ph_patch_2,
                                                           patch_3=ph_patch_3,
                                                           gt_v=ph_label,
                                                           channels=cfg.CHANNELS,
                                                           filters=cfg.FILTERS,
                                                           max_disparity=cfg.MAX_DISPARITY,
                                                           learning_rate=learning_rate,
                                                           global_step=global_step,
                                                           is_training=ph_is_training)   

    #run session
    saver = tf.train.Saver()
    sess = tf.Session(config=tf.ConfigProto())
    sess.run(tf.global_variables_initializer())
    #saver.restore(sess = sess, save_path= cfg.SAVE_PATH.format(cfg.TEST_ITER))
    training_losses, val_losses = [], []

    for i in range(40000):


        
        label, patch_2, patch_3 = get_random_patch(list_2=images_2,
                                                  list_3=images_3,
                                                  list_label=noc_1,
                                                  list_valid_pixels=valid_locations,
                                                  receptive_field_size=receptive_field,
                                                  max_disparity=cfg.MAX_DISPARITY,
                                                  batch_size=cfg.BATCH_SIZE,
                                                  loss_weights=cfg.LOSS_WEIGHTS)
   
        _ = sess.run(train_step, feed_dict={ph_patch_2:patch_2,
                                              ph_patch_3:patch_3,
                                              ph_label:label,
                                              ph_is_training:True})
        
        if i+1 % 100 == 0:
        
            loss_train, out = sess.run([cross_entropy, output_layer_t], feed_dict={ph_patch_2:patch_2,
                                                                              ph_patch_3:patch_3,
                                                                              ph_label:label,
                                                                              ph_is_training:False})

            label_val, patch_2_val, patch_3_val = get_random_patch(list_2=images_2_val,
                                                                    list_3=images_3_val,
                                                                    list_label=noc_1_val,
                                                                    list_valid_pixels=valid_locations_val,
                                                                    receptive_field_size=receptive_field,
                                                                    max_disparity=cfg.MAX_DISPARITY,
                                                                    batch_size=cfg.BATCH_SIZE,
                                                                    loss_weights=cfg.LOSS_WEIGHTS)

            loss_val = sess.run(cross_entropy, feed_dict={ph_patch_2:patch_2_val,
                             ph_patch_3:patch_3_val,
                             ph_label:label_val,
                             ph_is_training:False})

            print('Iter: ', i, 'Val loss: ', loss_val, 'Train loss: ', loss_train,
                  'Max prob: ',np.max(np.exp(-out[0])), 'Pred: ',np.argmax(np.exp(-out[0])), 'GT: ', np.argmax(label[0]))
            
            if i % 10000 == 0:
                saver.save(sess, cfg.SAVE_PATH.format(i))

def test_model():
    global_step = tf.Variable(0, trainable=False)

    boundaries = [24000, 32000]
    rates = [0.01, 0.01/5, 0.01/25]
    learning_rate = tf.train.piecewise_constant(global_step, boundaries, rates)

    image_2_path = cfg.KITTIPATH+'/training/image_2/000005_10.png'
    image_3_path = cfg.KITTIPATH+'/training/image_3/000005_10.png'
    disparity_image_path = cfg.KITTIPATH+'/training/disp_noc_0/000005_10.png'

    raw_image_2 = np.expand_dims(ndimage.imread(image_2_path, mode='RGB'), axis=0)
    raw_image_3 = np.expand_dims(ndimage.imread(image_3_path, mode='RGB'), axis=0)
    norm_image_2 = (raw_image_2-np.mean(raw_image_2))/np.std(raw_image_2)
    norm_image_3 = (raw_image_3-np.mean(raw_image_3))/np.std(raw_image_3)
    disparity_image = ndimage.imread(disparity_image_path, mode='I')/255
    
    # Initialize placeholders
    ph_patch_2, ph_patch_3, ph_label, ph_is_training = init_placeholders(raw_image_2.shape[1], raw_image_2.shape[2], cfg.CHANNELS, 0)

    # Get graph
    _, _, _, out_2, out_3 = build_graph(patch_2=ph_patch_2,
                                       patch_3=ph_patch_3,
                                       gt_v=ph_label,
                                       channels=cfg.CHANNELS,
                                       filters=cfg.FILTERS,
                                       max_disparity=cfg.MAX_DISPARITY,
                                       learning_rate=learning_rate,
                                       global_step=global_step,
                                       is_training=ph_is_training)

    #run session
    sess = tf.Session(config=tf.ConfigProto())
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess = sess, save_path= cfg.SAVE_PATH.format(cfg.TEST_ITER))
    
    vec_2, vec_3 = sess.run([out_2,out_3], feed_dict={ph_patch_2:norm_image_2,
                                                        ph_patch_3:norm_image_3,
                                                        ph_is_training:False})
    
    output = np.zeros([cfg.MAX_DISPARITY, vec_2.shape[1], vec_2.shape[2]])

    for i in tqdm(range(cfg.MAX_DISPARITY)):
            slice_2 = vec_2[:,:,i:vec_2.shape[2],:]
            slice_3 = vec_3[:,:,0:vec_2.shape[2]-i,:]
            inner_product = np.sum(np.multiply(slice_2, slice_3), axis=3)
            output[i,:,i:vec_2.shape[2]] = inner_product

    max_disp_index = np.argmax(output, axis=0)

    # convert to colour
    color_map = disparity_to_color(np.float32(max_disp_index))
    cmap = np.uint8(np.moveaxis(color_map, 0, 2)*128)
    color_image = Image.fromarray(cmap, 'RGB')
    color_image.save('tf_img_{}.png'.format(cfg.TEST_ITER))

    color_map = disparity_to_color(np.float32(disparity_image))
    cmap = np.uint8(np.moveaxis(color_map, 0, 2)*128)
    color_image = Image.fromarray(cmap, 'RGB')
    color_image.save('tf_img_gt_{}.png'.format(cfg.TEST_ITER))

    print('2px error: ', get_pixel_error(2, max_disp_index, disparity_image))
    print('3px error: ', get_pixel_error(3, max_disp_index, disparity_image))
    print('4px error: ', get_pixel_error(4, max_disp_index, disparity_image))
    print('5px error: ', get_pixel_error(5, max_disp_index, disparity_image))
    print('7px error: ', get_pixel_error(7, max_disp_index, disparity_image))
    print('10px error: ', get_pixel_error(10, max_disp_index, disparity_image))
    print('20px error: ', get_pixel_error(20, max_disp_index, disparity_image))
    print('30px error: ', get_pixel_error(30, max_disp_index, disparity_image))

    err = []
    for i in tqdm(range(128)):
        err.append(get_pixel_error(i, max_disp_index, disparity_image))

    plt.plot(err)
    plt.savefig('pxerror_{}.png'.format(cfg.TEST_ITER))
    
# some global variables
if __name__ == '__main__':
    receptive_field = 2 * (cfg.CONV_LAYERS * int(cfg.KERNEL/2)) + 1
    if args.mode == 'preprocess':
        preprocess()
    elif args.mode == 'train':
        train_model()
    elif args.mode == 'test':
        test_model()
    else:
        print('Mode not implemented. See --help.')
