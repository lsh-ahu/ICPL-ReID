import glob
import re
import os.path as osp
import os

from .bases_multi_modal import BaseImageDataset

class MSVR310(BaseImageDataset):
    
    dataset_dir = 'MSVR310'

    def __init__(self, root='', verbose=True, **kwargs):
        super(MSVR310, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query3')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        self._check_before_run()
        
        train = self._process_dir(self.train_dir, relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        if verbose:
            print("=> multi_modal_rgb loaded")
            self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery
        
        self.num_train_pids, self.num_train_imgs, self.num_train_cams, self.num_train_scenes = self.get_imagedata_info(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams, self.num_query_scenes = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams, self.num_gallery_scenes = self.get_imagedata_info(self.gallery)


    def _check_before_run(self):
        """Check if all files are available before going deeper"""
        if not osp.exists(self.dataset_dir):
            raise RuntimeError("'{}' is not available".format(self.dataset_dir))
        if not osp.exists(self.train_dir):
            raise RuntimeError("'{}' is not available".format(self.train_dir))
        if not osp.exists(self.query_dir):
            raise RuntimeError("'{}' is not available".format(self.query_dir))
        if not osp.exists(self.gallery_dir):
            raise RuntimeError("'{}' is not available".format(self.gallery_dir))

    def _process_dir(self, dir_path, relabel=False):
    
        RGBpath = osp.join(dir_path, 'RGB')
        NIpath = osp.join(dir_path, 'NI')
        TIpath = osp.join(dir_path, 'TI')
          
        ids_file = os.listdir(dir_path)
        
        img_paths_rgb= []
        
        for i in ids_file:
            path = glob.glob(osp.join(dir_path,i,'vis', '*.jpg'))
            
            img_paths_rgb.extend(path)

        pattern = re.compile(r'([-\d]+)_s(\d+)_v(\d+)_(\d+)')

        pid_container = set()
        
       
        for img_path in img_paths_rgb:

            # pid, _ = map(int, pattern.search(img_path).groups())
            pid, sid, vid ,index = map(int, pattern.search(img_path).groups())
            
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
       
        for img_path in img_paths_rgb:
            pid, sid, vid ,index= map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            #camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]

            rgb_path = img_path
            ni_path = img_path.replace("vis","ni")
            th_path = img_path.replace("vis","th")

            dataset.append((img_path,ni_path,th_path, pid, sid, vid))

        return dataset
