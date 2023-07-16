import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS


@DATASETS.register_module
class CocoLHCDataset(CustomDataset):
    CLASSES = ("Baby diapers",
               "Baby Furniture",
               "Baby washing and nursing supplie",
               "Baby slippers",          
               "Baby handkerchiefs ",
               "Baby crib",         
               "Baby carriage",
               "Baby tableware",
               "Dairy",
               "Cocktail",
               "Red wine",
               "Liquor and Spirits",
               "Carbonated drinks",
               "Herbal tea",
               "Coffee",
               "Tea beverage",
               "Baby milk powder",
               "Guozhen",
               "Ginger Tea",             # DeleteClass 20191102
               "Sour Plum Soup",            # DeleteClass 20191102  20
               "Adult milk powder",
               "Tea",
               "Notebook",
               "Pencil case",
               "Pen",
               "Baby Toys",
               "Children Toys",
               "Football",
               "Rubber ball",           # DeleteClass 20191103
               "Badminton",
               "Basketball",
               "Skate",
               "Pasta",
               "Noodle",
               "Flour",
               "Rise",
               "Oats",
               "Sesame paste",
               "Soymilk",
               "Lotus root flour",          # DeleteClass 20191103
               "Walnut powder",
               "Quick-frozen Tangyuan",
               "Quick-frozen Wonton",
               "Quick-frozen dumplings",
               "Can",
               "Instant noodles",
               "Mixed congee",
               "Potato chips",
               "Dried meat",
               #"Chicken claws",             # DeleteClass 20191103           20200114Del
               "Hot strips",             # DeleteClass 20191103
               "Dried fish",             # DeleteClass 20191103
               "Dried beans",             # DeleteClass 20191103
               "Fish tofu",             # DeleteClass 20191103
               "Chocolates",
               "Chewing gum",
               "Cake",
               "Pie",
               "Biscuits",
               # "Potatoes",    # DeleteClass 20191102           20200114Del
               "Ice cream",
               "Cooking wine",
               "Soy sauce",
               "Sauce",
               "Vinegar",
               "Care Kit",
               "Shampoo",
               "Hair conditioner",
               "Hair gel",
               "Hair dye",
               "Comb",
               "Tampon",
               "Cotton swab",
               "Band aid",
               "Adult Diapers",
               "Bath lotion",
               "Soap",                 # DeleteClass 20191102
               # "Flower dew",           # DeleteClass 20191102  10       20200114Del
               "Emulsion",             
               "Facial Cleanser",
               "Razor",
               "Facial mask",
               "Skin care set",
               "Toothbrush",
               # "Dental floss bar",        # DeleteClass 20191102       20200114Del
               "Toothpaste",
               "Mouth wash",
               "Makeup tools",
               "Jacket",
               "Trousers",
               "Adult shoes",
               "Adult socks",
               "Children shoes",
               "Children Socks",
               "Children hats",
               "Children underwear",             # DeleteClass 20191103
               "Lingerie",
               "Men underwear",
               "Adult hat",
               "Bedding set",
               "Juicer",
               "Washing machine",
               "Microwave Oven",
               "Desk lamp",
               "Air conditioning fan",    
               "Air conditioner",
               "Soybean Milk machine",
               "Electric iron",
               "Electric kettle",
               # "Pressure cooker",            # DeleteClass 20191102       20200114Del
               "Television",
               "Electric Hot pot",
               "Electric fan",
               "Rice cooker",
               "Electromagnetic furnace",
               "Electric frying pan",
               "Electric steaming pan",             # DeleteClass 20191103
               "Hair drier",                
               "Socket",
               "Refrigerator",
               "Coat hanger",
               "Sports cup",
               "Disposable cups",
               "Thermos bottle",
               "Basin",
               "Mug",
               "Draw bar box",
               "Trash",
               "Disposable bag",
               "Storage box",
               "Storage bottle",
               "Stool",     # NewClass20191010
               # "Package",     # DeleteClass 20191102       20200114Del
               "Knapsack",
               "Chopping block",
               "Food box",
               "Spoon",             # DeleteClass 20191103
               "Chopsticks",
               "Bowl",     # NewClass20191010
               "Pot shovel",
               "Soup ladle",
               "Cutter",             # DeleteClass 20191103
               # "Oil brush", #DeleteClass20191027
               "Knives",       # DeleteClass 20191102
               "Forks",
               "Dinner plate",
               "Fresh-keeping film"
               # "Fresh-keeping bag"   # DeleteClass 20191102               20200114Del
               )

    def load_annotations(self, ann_file):
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(self.img_infos[idx], ann_info)

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        return valid_inds

    def int2bin(self, Val):
        bin_list = [0,0,0,0,0,0,1]
        Valb = bin(Val)
        Valb = Valb[2:].zfill(7)
        bin_list2 = [int(e) for e in Valb]
        if len(bin_list2) == 7:
            return bin_list2
        return bin_list

    def int2bin3(self, Val):
        Val == int(Val)        
        if Val==1:
            return [0,0,1]
        elif 1<Val<5:
            return [0,1,0]
        elif 5<=Val<11:
            return [0,1,1]
        elif 11<=Val<16:
            return [1,0,0]
        elif 16<=Val<21:
            return [1,0,1]
        elif 21<=Val<30:
            return [1,1,0]
        elif Val>=30:
            return [1,1,1]

    def bin2int(self, bin_list):
        Valb=''
        for e in bin_list:
            Valb += e
        return int(Valb,2)


    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, seg_map. "masks" are raw annotations and not
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_counts = []
        gt_bboxes_ignore = []
        gt_masks_ann = []

        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]

            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_counts.append(ann['count'])   #Fixme: + count
                #gt_counts.append(self.int2bin3(ann['count']))   #Fixme: +  Resized binary count 1==>7
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann['segmentation'])

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
            gt_counts = np.array(gt_counts, dtype=np.int64)    #Fixme: + count
            #print('coco_LHC/gt_counts',gt_counts)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)
            gt_counts = np.array([], dtype=np.int64)    #Fixme: +  Resized binary count [] ==> [0,0,0,0,0,0,1] float32

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            counts=gt_counts,         #Fixme: + count
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann
