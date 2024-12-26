from STEGO.src.modules import *
# import hydra
from collections import defaultdict
import cv2
import numpy
import torch.multiprocessing
from PIL import Image
from STEGO.src.crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from STEGO.src.train_segmentation import LitUnsupervisedSegmenter
from STEGO.src.data import *
from tqdm import tqdm
import random
import copy
torch.multiprocessing.set_sharing_strategy('file_system')
# from util_borders import *
import gc


class UnlabeledImageFolder(Dataset):

    # def __init__(self, root, transform):
    def __init__(self, img, transform):
        super(UnlabeledImageFolder, self).__init__()
        # self.root = join(root)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img)
        self.root = img_pil
        self.transform = transform
        # self.images = os.listdir(self.root)
        self.images = img_pil

    def __getitem__(self, index):
        # image = Image.open(join(self.root, self.images[index])).convert('RGB')
        image = self.images
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        return image, "image"
        # return image, self.images[index]

    def __len__(self):
        # return len(self.images)
        return 1

#  enum  	cv::ColormapTypes {
#   cv::COLORMAP_AUTUMN = 0,
#   cv::COLORMAP_BONE = 1,
#   cv::COLORMAP_JET = 2,
#   cv::COLORMAP_WINTER = 3,
#   cv::COLORMAP_RAINBOW = 4,
#   cv::COLORMAP_OCEAN = 5,
#   cv::COLORMAP_SUMMER = 6,
#   cv::COLORMAP_SPRING = 7,
#   cv::COLORMAP_COOL = 8,
#   cv::COLORMAP_HSV = 9,
#   cv::COLORMAP_PINK = 10,
#   cv::COLORMAP_HOT = 11,
#   cv::COLORMAP_PARULA = 12,
#   cv::COLORMAP_MAGMA = 13,
#   cv::COLORMAP_INFERNO = 14,
#   cv::COLORMAP_PLASMA = 15,
#   cv::COLORMAP_VIRIDIS = 16,
#   cv::COLORMAP_CIVIDIS = 17,
#   cv::COLORMAP_TWILIGHT = 18,
#   cv::COLORMAP_TWILIGHT_SHIFTED = 19,
#   cv::COLORMAP_TURBO = 20,
#   cv::COLORMAP_DEEPGREEN = 21
# }
class Stego(Dataset):
  def __init__(self):
    self.color_ids = {}
    self.label_ids = {}
    self.id_labels = {}
    self.gc_count  = 0
    # self.colormap = create_cityscapes_colormap()
    # self.colormap_type = "cityscapes"
    self.colormap = None
    self.colormap_ids = []
    self.ranked_colormap = None
    self.ranked_colormap_num = None

  def report_gpu(self):
      print(torch.cuda.list_gpu_processes())

  # freeing cuda memory has bad side-effects 
  def free_cuda_memory(self):
      self.gc_count += 1
      print("garbach collection count", self.gc_count)
      # print(torch.cuda.list_gpu_processes())
      if self.gc_count > 15: 
        gc.collect()
        self.report_gpu()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        self.gc_count = 0

  def run(self, img, add_border=True):
    cfg = OmegaConf.load("do_stego_config.yml")
    result_dir = "./STEGO/results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "linear"), exist_ok=True)

    model = LitUnsupervisedSegmenter.load_from_checkpoint(cfg.model_path)
    # print(OmegaConf.to_yaml(model.cfg))
    # resized_img = cv2.resize(img, (356, 256))
    # gray= cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    resized_img = self.convert_plot_to_cv(img)
    if add_border:
      resized_img = self.pad_to_resize(resized_img, (416, 416))
    else:
      resized_img = cv2.resize(resized_img, (416, 416))
    np_img = numpy.asarray(resized_img)
    cv2.imshow('resized img', resized_img)
    # np_img = np_img.transpose(2,0,1)
    # np_img = np_img.reshape(1,3,416,416)

    dataset = UnlabeledImageFolder(
         # root=cfg.image_dir,
         img=np_img,
         # img=resized_img,
         # img=gray,
         # img=np_img,
         transform=get_transform(cfg.res, False, "center"),
    )

    # loader = DataLoader(dataset, cfg.batch_size * 2,
    #                   shuffle=False, num_workers=cfg.num_workers,
    #                   pin_memory=True, collate_fn=flexible_collate)
    loader = DataLoader(dataset, 1,
                        shuffle=False, num_workers=2,
                        pin_memory=True, collate_fn=flexible_collate)

    model.eval().cuda()
    if cfg.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net


    # tqdm provides a progress bar
    #
    # loader identifies the image’s location on disk, 
    # converts that to a tensor using read_image, 
    # retrieves the corresponding label from the csv data in self.img_labels,
    # calls the transform functions on them (if applicable),
    # and returns the tensor image and corresponding label in a tuple.
    #
    for i, (img, name) in enumerate(tqdm(loader)):
    # if True:
        with torch.no_grad():
            img = img.cuda()
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
            cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

            # fig, ax = plt.subplots(3, 1, figsize=(1 * 3, 3 * 3))
            for j in range(img.shape[0]):
                single_img = img[j].cpu()
                linear_crf = dense_crf(single_img, linear_probs[j]).argmax(0)
                cluster_crf = dense_crf(single_img, cluster_probs[j]).argmax(0)

            print("linear, clust", linear_crf.shape, cluster_crf.shape)

            # convert float cluster_crf into uint8 plot img
            plot_img = copy.deepcopy((cluster_crf*255).astype(np.uint8))
            # self.free_cuda_memory()
            return plot_img

        ###############################
        # Save the images in png format
        # Not currently used
        ###############################
            # new_name = ".".join(name[j].split(".")[:-1]) + ".png"
            # new_name = "image.png"
            # Image.fromarray(linear_crf.astype(np.uint8)).save(join(result_dir, "linear", new_name))
            # Image.fromarray(cluster_crf.astype(np.uint8)).save(join(result_dir, "cluster", new_name))

        ################################
        # Plot the predictions for paper
        ################################
            # saved_data = defaultdict(list)
            # saved_data["img"].append(img.cpu())
            # print("plot_img", plot_img.shape)  # was this correctly copied?
            # plot_img = numpy.squeeze(plot_img) # was this correctly copied?
            # print("plot_img0", plot_img.shape)
            # plot_img = ((saved_data["img"][0]) * 255).unsqueeze(0).numpy().astype(np.uint8)
            # if cfg.run_prediction:
            #     plot_cluster = (model.label_cmap[
            #         model.test_cluster_metrics.map_clusters(
            #             plot_img)]).astype(np.uint8)
            #     plot_cluster = (model.label_cmap[
            #         model.test_cluster_metrics.map_clusters(
            #             saved_data["cluster_preds"][img_num])]) \
            #         .astype(np.uint8)
            # ax[1].imshow(plot_cluster)
            # ax[0].set_ylabel("Image", fontsize=26)
            # if cfg.run_prediction:
            #     ax[2].set_ylabel("STEGO\n(Ours)", fontsize=26)
            # remove_axes(ax)
            # plt.tight_layout()
            # plt.show()
            # plt.clf()

  def pad_to_resize(self, img, new_sz):
      old_sz = img.shape
      resized_img = cv2.copyMakeBorder(img,
           top=0, bottom=(new_sz[0] - old_sz[0]), 
           left=0, right=(new_sz[1] - old_sz[1]),
           borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])
      # cv2.imshow("Resized with buffer", resized_img)
      # cv2.waitKey(0)
      return resized_img
 


  def convert_cv_to_plot(self, cv_img):
      # RGB_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
      blue = RGB_image[:,:,0]
      green = RGB_image[:,:,1]
      red = RGB_image[:,:,2]
      plot_img = blue/3+green/3+red/3
      plot_img = np.asarray(plot_img, dtype=int)
      # plot_img = img.astype(np.float32)
      # plot_img /= 255.
      # plot_img = Image.fromarray( np.asarray( np.clip(cv_img,0,255), dtype="uint8"), "L" )
      return plot_img

  def convert_plot_to_rand_cv(self, plot_img):
      cv_plot_img = np.zeros((len(plot_img), len(plot_img[0]), 3), dtype="uint8")
      num_color = 0
      # TODO: convert to use vector operations
      for i,row in enumerate(plot_img):
        for j,pix in enumerate(row):
          try:
            # pix is a classification key, each of which we map to random color
            hash_nm = (pix[0]) + (256*pix[1]) + (256*256*pix[2])
            # print("pix:", pix, hash_nm)
            cv_plot_img[i][j] = self.color_ids[hash_nm]
          except:
            hash_nm = (pix[0]) + (256*pix[1]) + (256*256*pix[2])
            # print("pix:", pix, hash_nm)
            # print("color_ids:", self.color_ids)
            self.color_ids[hash_nm] = (random.randint(0, 255),random.randint(0, 255),random.randint(0, 255))
            # print("post pix")
            cv_plot_img[i][j] = self.color_ids[hash_nm]
      # print("unique colors:", self.color_ids.keys())
      return cv_plot_img

  def convert_plot_to_cv(self, plot_img):
      # pil_image = Image.open('Image.jpg').convert('RGB')
      # open_cv_image = numpy.array(pil_image)
      open_cv_image = numpy.array(plot_img)
      # Convert RGB to BGR
      open_cv_image = open_cv_image[:, :, ::-1].copy()
      return open_cv_image

  # It's possible to do a faster version of covert_to_high_contrast
  # not completed but obsolete anyway; using Stego implementation colormaps.
  def convert_to_high_contrast_stego_img_fast_old(self, stego_img):
      stgo = np.array(stego_img)
      id, id_cnts = np.unique(stgo, return_counts=True)
      cnts_ids = {k: v for k, v in sorted(id_cnts.items(), key=lambda item: item[1], reverse=True)}
      print("id_cnts", id_cnts)
      nice_stego = np.ones_like(stgo)*255
      num_ids = len(id_cnts)
      # NO_LONGER_TODO: CONVERT THE REST TO USE ARRAY INTERFACE

  def int_to_bgr_256(self, rgbint):
      blue =  rgbint & 255
      green = (rgbint >> 8) & 255
      red =   (rgbint >> 16) & 255
      print("int_to_rgb: ", rgbint, blue, green, red)
      return (blue,green,red)

  # relatively fast; maintains state to minimize duplicate work; uses nparray.
  def create_high_contrast_colormap_256(self, img):
      if self.ranked_colormap is None:
        numcolors = 256 * 256 * 256  # rgb 
        self.ranked_colormap = [0, numcolors-1]
        todo_lst = [[0, numcolors-1]]
        while len(self.ranked_colormap) < 256:
          if len(todo_lst) == 0:
            break
          self.split_lst(self.ranked_colormap, todo_lst)
        print("ranked_colormap:", self.ranked_colormap)
      if self.colormap is None:
        self.colormap = np.zeros((512, 3), dtype=np.uint8)
        self.colormap_ids = []
        # total number of pixels for each stego segment id
        unique_vals,counts = np.unique(img, return_counts=True) 
        counts = np.array(counts)
        order = counts.argsort()
        ranks = order.argsort()
        print("unique:", unique_vals)
        print("counts:", counts)
        print("order :", order )
        print("ranks :", ranks)
# Examples:
# unique: [  0   231  233 236   240 241  242 247 248 250 251 255]
# counts: [11166 23605 13 75734 782 3132 55333 2 2959 45 267 18]
# order : [  7    2    11  9     10  4     8  5   0   1   6   3]
# ranks : [  8    9     1 11      5  7    10  0   6   3   4   2]
#            0    1     2  3      4  5     6  7   8   9  10  11
#
# colormap[unique_vals[i]]  =  int_to_rgb(self.ranked_colormap[ranks[i]))
#

        for i, r in enumerate(ranks):
          self.colormap[unique_vals[i]] = self.int_to_bgr(self.ranked_colormap[r])
        print("colormap:", self.colormap)
      else:
        # new IDs that need to be added to the colormap (appended in rank order)
        unique_vals,counts = np.unique(img, return_counts=True) 
        counts = np.array(counts)
        order = counts.argsort()
        ranks = order.argsort()
        new_vals = list(np.setxor1d(np.array(unique_vals), self.colormap_ids))
        for i, r in enumerate(ranks):
          if unique_vals[i] not in new_vals:
            self.colormap[unique_vals[i]] = self.int_to_bgr(self.ranked_colormap[r])

#Low       0    0    0
#6        85   85    85
#1        85    0    0
#2         0   85    0
#3         0    0    85
#4        85   85    0
#5         0   85    85
#1        42    0     0
#          0   42     0
#          0    0    42
#          0   42    42
#         43   42     0
#6        42    0    42
#6        21   x 6
#6        63   x 6

# Low High
# [0 85]
# [0 42]
# [42 85]
# [0 21]
# [21 85]
# [64 0]


  def color_permutations(self, min_nm, max_nm, new_min=False, new_max=False):
      perm = []
      if new_min:
        perm.append((min_nm, min_nm, min_nm))
      if new_max:
        perm.append((max_nm, max_nm, max_nm))
      perm.append((max_nm, min_nm, min_nm))
      perm.append((min_nm, max_nm, min_nm))
      perm.append((min_nm, min_nm, max_nm))
      # Will the following be too similar to its predecessor?
      # perm.append((max_nm, max_nm, min_nm))
      # perm.append((max_nm, min_nm, max_nm))
      # perm.append((min_nm, max_nm, max_nm))
      return perm

  def split_lst(self, rank_lst, todo_lst):
      min_nm, max_nm = todo_lst[0]
      mid_nm =  (min_nm + max_nm) // 2
      if (max_nm - mid_nm > 1):
        todo_lst.append([mid_nm, max_nm])
      if (mid_nm - min_nm > 1):
        todo_lst.append([min_nm, mid_nm])
      rank_lst.append(mid_nm)
      self.ranked_colormap += self.color_permutations(0, max_nm, new_min=False, new_max=True)
      self.ranked_colormap_num.append(mid_nm)
      todo_lst.pop(0)

  def int_to_bgr(self, rgbint):
      blue =  rgbint % 85
      green = (rgbint % 170) - (rgbint % 85)
      red =  rgbint - (rgbint % 170)
      print("int_to_rgb: ", rgbint, blue, green, red)
      # mult 85 by 3 to revert to 256
      return (blue*3,green*3,red*3)

  # relatively fast; maintains state to minimize duplicate work; uses nparray.
  def create_high_contrast_colormap(self, img):
      if self.ranked_colormap is None:
        num_colors = 85 + 85 + 85
        todo_lst = [[0, num_colors]]
        self.ranked_colormap_num = [0, 85]
        self.ranked_colormap = self.color_permutations(0, num_colors, new_min=True, new_max=True)
        while len(self.ranked_colormap) < 256:
          if len(todo_lst) == 0:
            break
          self.split_lst(self.ranked_colormap, todo_lst)
        print("ranked_colormap:", self.ranked_colormap)
        print("ranked_colormap_num:", self.ranked_colormap_num)
      if self.colormap is None:
        self.colormap = np.zeros((512, 3), dtype=np.uint8)
        self.colormap_ids = []
        # total number of pixels for each stego segment id
        unique_vals,counts = np.unique(img, return_counts=True) 
        counts = np.array(counts)
        order = counts.argsort()
        ranks = order.argsort()
        print("unique:", unique_vals)
        print("counts:", counts)
        print("order :", order )
        print("ranks :", ranks)
        # Examples:
        # unique: [  0   231  233 236   240 241  242 247 248 250 251 255]
        # counts: [11166 23605 13 75734 782 3132 55333 2 2959 45 267 18]
        # order : [  7    2    11  9     10  4     8  5   0   1   6   3]
        # ranks : [  8    9     1 11      5  7    10  0   6   3   4   2]
        #            0    1     2  3      4  5     6  7   8   9  10  11
        #
        # colormap[unique_vals[i]] = int_to_rgb(self.ranked_colormap[ranks[i]))
        #
        for i, r in enumerate(ranks):
          self.colormap[int(unique_vals[i])] = self.ranked_colormap[r]
          # self.colormap[unique_vals[i]] = self.int_to_bgr(self.ranked_colormap[r])
        print("colormap:", self.colormap)
      else:
        # new IDs that need to be added to the colormap (appended in rank order)
        unique_vals,counts = np.unique(img, return_counts=True) 
        counts = np.array(counts)
        order = counts.argsort()
        ranks = order.argsort()
        new_vals = list(np.setxor1d(np.array(unique_vals), self.colormap_ids))
        for i, r in enumerate(ranks):
          if unique_vals[i] not in new_vals:
            self.colormap[int(unique_vals[i])] = self.int_to_bgr(self.ranked_colormap[r])

  def convert_to_high_contrast_stego_img(self, stego_img):
      self.create_high_contrast_colormap(stego_img)
      return self.colormap[stego_img]
      return cv2.applyColorMap(stego_img, cv2.COLORMAP_RAINBOW)

  def convert_to_high_contrast_stego_img_slow(self, stego_img):
      # total number of pixels for each stego segment id
      id_cnts = {}
      id_x = {}
      id_y = {}
      segment = {}
      # for x in range(len(stego_img)):
        # for y in range(len(stego_img[0])):
      for x,row in enumerate(stego_img):
        for y,id  in enumerate(row):
          # id = stego_img[x][y]
          # print("id:", id, x, y)
          try:
            id_cnts[id] += 1
            id_x[id].append(x)
            id_y[id].append(y)
            segment[id][x][y] = max(stego_img[x][y], 255)
          except:
            id_cnts[id] = 1
            id_x[id] = [x]
            id_y[id] = [y]
            segment[id] = np.zeros((len(stego_img), len(stego_img[0]), 1), dtype="uint8")
      # sort the id_cnts in order of counts
      cnts_ids = {k: v for k, v in sorted(id_cnts.items(), key=lambda item: item[1], reverse=True)}
      print("id_cnts", id_cnts)
      nice_stego = np.ones_like(stego_img)*255
      num_ids = len(id_cnts.keys())
      print("nice_stego: num_ids:", num_ids, 255/num_ids)
      f = 0
      b = num_ids-1
      id_val = {}
      id_val[id] = int(255 / (num_ids-1) * f)

      for id_num, id in enumerate(cnts_ids.keys()):
        if id_num % 2:
          id_val[id] = int(255 / (num_ids-1) * f)
          f += 1
        else:
          id_val[id] = int(255 / (num_ids-1) * b)
          b -= 1
        # print("nice_stego: id, vals:", id, id_val[id], id_cnts[id], id_num, f, b)
      for x,row in enumerate(stego_img):
        for y,id  in enumerate(row):
          nice_stego[x][y] = 255 - id_val[id]
      return nice_stego

  # allow detection of tiny squares
  def find_polygon(self, img):
      # already done by find_cube: # convert the stitched image to grayscale and threshold it
      # such that all pixels greater than zero are set to 255
      # (foreground) while all others remain 0 (background)
      for obj_key, obj_color in self.color_ids.items():
        obj_img = img.copy()
        print("obj_color", obj_color)
        print("obj_img[0][0]", obj_img[0][0])
        print("l0,l00:",len(obj_img), len(obj_img[0]), len(obj_img[0][0]))
        tot_num_pix_in_obj = 0
        for i in range(len(obj_img)):
          for j in range(len(obj_img[0])):
            if (obj_img[i][j][0] != obj_color[0] or
                obj_img[i][j][1] != obj_color[1] or
                obj_img[i][j][2] != obj_color[2]):
              obj_img[i][j] = [255,255,255]
            else:
              obj_img[i][j] = [0,0,0]
              tot_num_pix_in_obj += 1
        cv2.imshow("stego obj_img", obj_img)

        shape, approximations = None, None
        squares = []
        # find all external contours in the threshold image then find
        # the *largest* contour which will be the contour/outline of
        # the stitched image
        try:
          sqimg = cv2.cvtColor(obj_img, cv2.COLOR_BGR2GRAY)
        except:
          sqimg = obj_img
        sqimg = cv2.bitwise_not(sqimg)
        imagecontours, hierarchy = cv2.findContours(sqimg,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        # for each of the contours detected, the shape of the contours is approximated 
        # using approxPolyDP() function and the contours are drawn in the image using 
        # drawContours() function
        # For our border case, there may be a few dots or small contours that won't
        # be considered part of the border.
        # print("real_map_border count:", len(imagecontours))
        if len(imagecontours) > 1:
          print("hierarchy:", hierarchy)
          for i, c  in enumerate(imagecontours):
            area = cv2.contourArea(c)
            M = cv2.moments(c)
            print(i, "area, moment:", area, M, len(c))
            print(i, "area:", area, len(c))
            if M["m00"] != 0:
              centroid_x = int(M["m10"] / M["m00"])
              centroid_y = int(M["m01"] / M["m00"])
              print(i, "centroid:", centroid_x, centroid_y)
        for count in imagecontours:
          area = cv2.contourArea(count)
          if area < 16:
            continue
          # epsilon = 0.01 * cv2.arcLength(count, True)
          epsilon = 0.01 * cv2.arcLength(count, True)
          approximations = cv2.approxPolyDP(count, epsilon, True)
          # e.g. [[[224 224]] [[252 372]] [[420 372]] [[447 224]]]
          #the name of the detected shapes are written on the image
          #
          # sqimg2 = sqimg.copy()
          sqimg2 = np.zeros_like(sqimg)
          cv2.drawContours(sqimg2, count, -1, 255, cv2.FILLED)
          # cv2.drawContours(sqimg2, count, -1, 255, 5)
          # sqimg2 = np.ones_like(sqimg)*255
          cv2.imshow("contours", sqimg2)
          print(cv2.contourArea(count), cv2.arcLength(count,True))
          if (cv2.contourArea(count) < cv2.arcLength(count,True)):
            print("closed contour")
          bb_x,bb_y,bb_w,bb_h = cv2.boundingRect(count)
          num_pix_in_obj = 0
          for w in range(bb_w):
            for h in range(bb_h):
              x = int(bb_x+w)
              y = int(bb_y+h)
              if obj_img[y][x][0] == 0:
                num_pix_in_obj += 1
          # num_pix_in_obj = cv2.countNonZero(obj_img)  
          sq_area = bb_w*bb_h
          # cv2.rectangle(sqimg2,(bb_x,bb_y),(bb_x+bb_w,bb_y+bb_h),(0,255,0),2)
          #####
          print("bounding box", num_pix_in_obj, (bb_x, bb_y), (bb_w, bb_h), num_pix_in_obj/sq_area)
          rect = cv2.minAreaRect(count)
          rot_rect_area = rect[1][0] * rect[1][1]
          box = cv2.boxPoints(rect)
          box = np.int0(box)
          # cv2.drawContours(img,[box],0,(0,0,255),2)
          print("rotated bounding box", num_pix_in_obj, rect, num_pix_in_obj/rot_rect_area)
          #####
          (circ_x,circ_y),radius = cv2.minEnclosingCircle(count)
          center = (int(circ_x),int(circ_y))
          radius = int(radius)
          circ_area = np.pi * radius * radius
          print("bounding circle", center, radius, num_pix_in_obj/circ_area)
          print("area", tot_num_pix_in_obj, num_pix_in_obj, sq_area, rot_rect_area, circ_area)
          # cv2.circle(img,center,radius,(0,255,0),2)
          cv2.waitKey(30)
          # hough_lines_image = np.ones_like(sqimg2)*255
          hough_image = np.zeros_like(sqimg2)
          hough_lines = self.get_hough_lines(sqimg2)
          print("hough_lines", hough_lines)
          if hough_lines is not None:
            print("new houghline")
            for line in hough_lines:
              # print("line", line)
              for x1,y1,x2,y2 in line:
                print( x1,y1,x2,y2)
                cv2.line(hough_image,(x1,y1),(x2,y2),255,5)
                # cv2.line(hough_image,(x1,y1),(x2,y2),0,5)
            cv2.imshow("stego lines obj_img", hough_image)
          if True:
            hough_image = np.zeros_like(sqimg2)
            circles = cv2.HoughCircles(sqimg2,cv2.HOUGH_GRADIENT,1,20,
                           param1=50,param2=30,minRadius=0,maxRadius=0)
            if circles is not None:
              circles = np.uint16(np.around(circles))
              print("circles", circles)
              for i in circles[0,:]:
                # draw the outer circle
                cv2.circle(hough_image,(i[0],i[1]),i[2],255,2)
              cv2.imshow("stego circles obj_img", hough_image)
            else:
              print("circles None")
          if cv2.isContourConvex(count):
            print("Convex Contour")
          if False:
            hough_image = np.zeros_like(sqimg2)
            convhull = cv2.convexHull(count)
            # cv2.drawContours( hough_image, convhull, (int)0, 255 )
            cv2.drawContours( hough_image, convhull, color=255 )
            cv2.imshow("stego hull obj_img", hough_image)

          cv2.waitKey(0)
          i, j = approximations[0][0] 
          if len(approximations) == 3:
            shape = "Triangle"
          elif len(approximations) == 4:
            shape = "Trapezoid"
            area = cv2.contourArea(approximations)
            # sqimg2 = sqimg.copy()
            # cv2.drawContours(sqimg2, count, -1, (0,255,0), 3)
            # cv2.imshow("contours", sqimg2)
            # cv2.waitKey()
            # if area > 100 &  cv2.isContourConvex(approximations):
            if cv2.isContourConvex(approximations):
              maxCosine = -100000000000000
              for j in range(2, 5):
                cosine = abs(middle_angle(approximations[j%4][0], approximations[j-2][0], approximations[j-1][0]))
                print("cosine ", cosine, ":", approximations[j%4][0], approximations[j-2][0], approximations[j-1][0])
                maxCosine = max(maxCosine, cosine)
              # if cosines of all angles are small
              # (all angles are ~90 degree) then write quandrange
              # vertices to resultant sequence
              if maxCosine < 0.3 and maxCosine >= 0:
                shape = "Square"
                squares.append(approximations)
                print("square found:", approximations)
              else:
                print("maxCos:", maxCosine)
            else:
              print("non convex contour", approximations)
          elif len(approximations) == 5:
            shape = "Pentagon"
          elif 6 < len(approximations) < 15:
            shape = "Ellipse"
          else:
            shape = "Circle"
          # if len(imagecontours) > 1:
          #   cv2.putText(thresh,shape,(i,j),cv2.FONT_HERSHEY_COMPLEX,1,0,2)
          #   cv2.waitKey(0)
          # print("map shape:", shape, approximations)
          #displaying the resulting image as the output on the screen
          # imageread = mapimg.copy()
          # print("contour:", count)
          # print("approx contour:", approximations)
          # return shape, approximations
          print("shape:", shape, squares)
        # return squares
      return squares

  def map_line_to_segment(self, persist_line, gripper_img):
      # map to the segment from STEGO, make BB
      pl_last_img_line = {}
      # ARD: eventually fix the nesting of the ds_num loops
      # for ds_num, ds in enumerate(dataset):
      #   for pass_num in range(num_passes):
      for pl_ds_num in range(num_ds):
        pl_num_imgs = len(dataset[pl_ds_num])
        for pl_img_num in range(pl_num_imgs):
          bb = []
          bb_maxw, bb_minw, bb_maxh, bb_minh = -1, 10000, -1, 10000
          for pl_num in range(len(persist_line)):
            pl_stats = persist_line_stats[pl_num]
            if pl_stats is not None:
              [mean_disp_x, mean_disp_y, stddev_disp_x, stddev_disp_y, mean_x, mean_y, mean_line, mean_line_length, mean_angle, counter, [pl_min_img_num, pl_max_img_num]] = pl_stats
              print("PL", pl_num, counter, mean_line, pl_min_img_num, pl_max_img_num)
            else:
              continue
            if mean_y > MIN_GRIPPER_Y:
              continue
            if counter < mean_counter:
              continue
            a_line = None
            try:
              print("pl angle_list:")
              angle_list = persist_line[pl_num][(pl_ds_num, pl_img_num)]
              # print(angle_list)
              l_maxw, l_minw, l_maxh, l_minh = -1, 10000, -1, 10000
              for a_num, pl_angle in enumerate(angle_list):
                asd_item = angle_sd[pl_angle]
                a_ds_num, a_img_num, a_line = asd_item
                l_maxw = max(a_line[0][0], a_line[0][2], l_maxw)
                l_minw = min(a_line[0][0], a_line[0][2], l_minw)
                l_maxh = max(a_line[0][1], a_line[0][3], l_maxh)
                l_minh = min(a_line[0][1], a_line[0][3], l_minh)
                if l_maxh > MIN_GRIPPER_Y:
                  l_maxh = MIN_GRIPPER_Y
              pl_last_img_line[pl_num] = [l_maxw, l_minw, l_maxh, l_minh]
            except:
              print("except pl angle_list:")
              try:
                [l_maxw, l_minw, l_maxh, l_minh] = pl_last_img_line[pl_num]
              except:
                continue
            if l_maxw == -1 or l_maxh == -1:
              print("skipping PL", pl_num)
              continue
#         bb = make_bb(bb_maxw, bb_minw, bb_maxh, bb_minh)
#         img_path = img_paths[pl_ds_num][pl_img_num]
#         img = cv2.imread(img_path)
#         bb_img = get_bb_img(img, bb)
#         print(pl_img_num, "bb", bb)
#         cv2.imshow("bb", bb_img)
#         # cv2.waitKey(0)

  # assume: for constructing "unmoved_pixels" like a gripper in a set position
  def set_pixel_prob(self, stego_plot_img, stego_pixel_prob=None):
      if stego_pixel_prob is None:
        stego_pixel_prob = [] 
        for i in range(len(stego_plot_img)):
          stego_pixel_prob.append([])
          for j in range(len(stego_plot_img[0])):
            # stego_pixel_prob[i][j] = {}
            stego_pixel_prob[i].append({})
      # print("set_pixel_prob", len(stego_plot_img), len(stego_plot_img[0]))
      for i in range(len(stego_plot_img)):
        for j in range(len(stego_plot_img[0])):
          rgb = stego_plot_img[i][j]
          id = (rgb[0]<<16)+(rgb[1]<<16)+(rgb[2])
          try:
            stego_pixel_prob[i][j][id] += 1
            # print("stego_pixel_prob[i][j][id]", stego_pixel_prob[i][j][id])
          except:
            # print("SET PIX PROB EXCEPT", id)
            stego_pixel_prob[i][j][id] = 1
      return stego_pixel_prob

  def get_unmoved_stego_plot_img(self, stego_pixel_prob, stego_img):
      stego_plot_img = np.zeros((len(stego_img), len(stego_img[0])), dtype="uint8")
      max_cnt = 0
      for rgb,cnt in stego_pixel_prob.items():
        id = (rgb[0]<<16)+(rgb[1]<<16)+(rgb[2])
        if cnt > max_cnt:
          max_cnt = cnt
          max_id = id
      for i in range(len(stego_img)):
        for j in range(len(stego_plot_img[0])):
          if stego_img[i][j] == max_id:
            stego_plot_img[i][j] = max_id
      return stego_plot_img

  # total number of pixels for each stego segment id
  def get_key_counts(self, stego_plot_img):
      id_cnts = {}
      for i in range(len(stego_plot_img)):
        for j in range(len(stego_plot_img[0])):
          rgb = stego_plot_img[i][j] 
          id = (rgb[0]<<16)+(rgb[1]<<16)+(rgb[2])
          try:
            id_cnts[id] += 1
          except:
            id_cnts[id] = 1
      return id_cnts

  def set_label(stego_key, label):
      try:
        stego_label, cnt = self.label_ids[stego_key]
        cnt += 1
        self.label_ids[stego_key] = [label, cnt]
      except:
        self.label_ids[stego_key] = [label, 1]
      try:
        key, cnt = self.id_labels[stego_key]
        cnt += 1
        self.id_labels[label] = [stego_key, cnt]
      except:
        self.id_labels[label] = [stego_key, 1]

  def get_labels(stego_key):
      total = 0
      for label, cnt in self.label_ids[stego_key].items():
        total += cnt
      label_prob = {}
      highest_prob_label = None
      highest_prob = 0
      for label, cnt in self.label_ids[stego_key].items():
        label_prob[label] = cnt/total
        if label_prob[label] > highest_prob:
          highest_prob_label = label
          highest_prob = label_prob[label]
      return highest_prob_label, label_prob
       
  def get_label_keys(label):
      total = 0
      for key, cnt in self.id_labels[label].items():
        total += cnt
      key_prob = {}
      for key, cnt in self.id_labels[label].items():
        key_prob[key] = cnt/total
      highest_prob_label = None
      highest_prob = 0
      for label, cnt in self.label_ids[stego_key].items():
        label_prob[stego_key] = cnt/total
        if label_prob[stego_key] > highest_prob:
          # Note: to find highest prob or relative prob, don't need to compute total
          highest_prob_key = label
          highest_prob = label_prob[label] 
      return highest_prob_key, key_prob

  # Take the bounding boxes and apply them to the STEGO images to get pixel boundaries.
  # find the % of the Stego image/pixels associated with each bb.
  # set STEGO label
  def analyze_bb(self, lg_bb, rg_bb, safe_ground_bb, stego_img):
      def get_cnt_bb(bb,img,keycnt):
          bb_maxw, bb_minw, bb_maxh, bb_minh = get_min_max_borders(bb)
          for w in range(bb_minw, bb_maxw):
            for h in range(bb_minh, bb_maxh):
              key = stego_img[w][h]
              try:
                keycnt[key] += 1
              except:
                keycnt[key] = 0
          return keycnt


      gripper_stego_keycnt = {}
      sg_stego_keycnt = {}
      key_cnt = self.get_key_counts(stego_img)
      gripper_stego_keycnt = get_cnt_bb(lg_bb,stego_img,gripper_stego_keycnt)
      gripper_stego_keycnt = get_cnt_bb(rg_bb,stego_img,gripper_stego_keycnt)
      stego_gripper_img = self.get_unmoved_stego_plot_img(gripper_stego_keycnt, stego_img)

      stego_sg_img = None
      if safe_ground_bb is not None:
        sg_stego_keycnt = get_cnt_bb(safe_ground_bb,stego_img,sg_stego_keycnt)
        for key, cnt in sg_stego_keycnt.items():
          if key_cnt[key]*.8 < cnt:
            print("gripper key:", key)
        stego_sg_img = self.get_unmoved_stego_plot_img(sg_stego_keycnt, stego_img)
  
      self.label_ids = {}
      return stego_gripper_img, stego_sg_img


  # when goto cube, when goto box, then forward should be safe ground.
  #   - call after forward
  # when arm moves for pickup, then could call too?
  # Other side of table boundary might be out of bounds. compare to edge lines.
  # Find rectangular shape.  Look for vanishing point.
  # multiple segments could be part of same table.
  # combine with track_objs.py.
  def analyze_table(self, stego_img, lines):
      pass

  def out_of_bounds(self, stego_img, segment_id):
      pass

  # 
  def analyze_box(self, lg_bb, rg_bb, stego_img):
      pass

  def analyze_cube(self, lg_bb, rg_bb, stego_img):
      pass

# if __name__ == "__main__":
#     prep_args()
#     my_app()
