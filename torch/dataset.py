import glob
import numpy as np
import pickle
import torch
import os

from collections import defaultdict
from nltk.tokenize import RegexpTokenizer
import torchvision.transforms as transforms

from PIL import Image


class CocoOneCategoryDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, words_num=18, captions_num=5, img_size=256, transform=None):
        self.data_path = data_path
        self.words_num = words_num
        self.captions_num = captions_num
        self.img_size = img_size
        self.transform = transform
        self.normalize = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        )

        self.load_text_data()
        self.tokenize_all_captions()
        self.make_word_to_idx()
        self.divide_class()
        self.drop_small_class()

    def __len__(self):
        return len(self.filenames)

    def load_text_data(self):
        filepath = os.path.join(self.data_path, "train_one_coco.pkl")
        with open(filepath, "rb") as f:
            self.captions_table = pickle.load(f)
            self.filenames = list(self.captions_table.keys())

    def tokenize_all_captions(self):
        for filename in self.captions_table:
            captions = self.captions_table[filename][1]
            tokenized_captions = []
            for cap in captions:
                if len(cap) == 0:
                    continue
                cap = cap.replace("\ufffd\ufffd", " ")
                tokenizer = RegexpTokenizer(r"\w+")
                tokens = tokenizer.tokenize(cap.lower())

                if len(tokens) == 0:
                    print("cap", filename)
                    continue

                tokens_new = []
                for t in tokens:
                    t = t.encode("ascii", "ignore").decode("ascii")
                    if len(t) > 0:
                        tokens_new.append(t)
                tokenized_captions.append(tokens_new)
            self.captions_table[filename][1] = tokenized_captions
    
    def make_word_to_idx(self):
        word_counts = defaultdict(float)
        for filename in self.captions_table.keys():
            captions = self.captions_table[filename][1]
            for cap in captions:
                for word in cap:
                    word_counts[word] += 1
        
        vocab = [w for w in word_counts if word_counts[w] >= 0]
        self.vocab_size = len(vocab)

        idxtoword = {}
        idxtoword[0] = "<end>"
        wordtoidx = {}
        wordtoidx["<end>"] = 0
        idx = 1
        for w in vocab:
            wordtoidx[w] = idx
            idxtoword[idx] = w
            idx += 1

        for filename in self.captions_table.keys():
            captions = self.captions_table[filename][1]
            new_captions = []
            for cap in captions:
                new_cap = [] 
                for word in cap:
                    new_cap.append(wordtoidx[word])
                new_captions.append(new_cap)
            self.captions_table[filename][1] = new_captions
    
    def divide_class(self):
        self.categories = {}
        total_categories = self.captions_table.keys()
        for filename in self.filenames:
            category = self.captions_table[filename][0]
            captions = self.captions_table[filename][1]
            if category in self.categories.keys():
                self.categories[category].append(filename)
            else :
                self.categories[category] = [filename]
    
    def drop_small_class(self):
        drop_classes = []
        for key in self.categories.keys():
            if len(self.categories[key]) < 100:
                drop_classes.append(key)
        
        drop_files = []
        for filename in self.captions_table.keys():
            if self.captions_table[filename][0] in drop_classes:
                drop_files.append(filename)
        
        for filename in drop_files:
            self.captions_table.pop(filename)
            self.filenames.remove(filename)
                    
    def get_caption(self, filename, sent_idx):
        captions = self.captions_table[filename][1]
        caption = np.asarray(captions[sent_idx]).astype("int64")
        
        num_words = len(caption)
        cap = np.zeros(self.words_num, dtype="int64")
        
        if num_words <= self.words_num:
            cap[:num_words] = caption
            cap_len = num_words
        else :
            idx = list(np.arange(num_words))
            np.random.shuffle(idx)
            idx = idx[: self.words_num]
            idx = np.sort(idx)
            cap[:] = caption[idx]
            cap_len = self.words_num

        return cap, cap_len

    def get_img(self, filename):
        img_path = self.data_path + '/coco_sample/' + filename
        img = Image.open(img_path).convert("RGB")
        width, height = img.size
        
        if self.transform is not None:
            img = self.transform(img)
            img = img.crop([0, 0, 256, 256])
        
        if self.normalize is not None:
            img = self.normalize(img) 
            
        return img

    def __getitem__(self, index):
        idx1 = index
        filename1 = self.filenames[idx1]
        category = self.captions_table[filename1][0]

        category_size = len(self.categories[category])

        idx2 = 0
        while self.categories[category][idx2] == filename1:
            idx2 = np.random.randint(0, category_size)

        filename2 = self.categories[category][idx2]

        img1 = self.get_img(filename1)
        img2 = self.get_img(filename2)
        
        source_captions_num = len(self.captions_table[filename1][1])
        target_captions_num = len(self.captions_table[filename2][1])

        source_cap_idx = np.random.randint(0, source_captions_num)
        source_cap, source_cap_len = self.get_caption(filename1, source_cap_idx)

        intra_cap_idx = source_cap_idx
        while intra_cap_idx == source_cap_idx:
            intra_cap_idx = np.random.randint(0, source_captions_num)
        intra_cap, intra_cap_len = self.get_caption(filename1, intra_cap_idx)

        inter_cap_idx = np.random.randint(0, target_captions_num)
        inter_cap, inter_cap_len = self.get_caption(filename2, inter_cap_idx)

        return img1, img2, source_cap, intra_cap, inter_cap 


if __name__=='__main__':
    image_transform = transforms.Compose([
        transforms.Scale(int(256 * 76 / 64))])
    dataset = CocoOneCategoryDataset('../data/', transform=image_transform)
    # dataset.divide_class()
    # dataset.drop_small_class()
    
    # for key in dataset.captions.keys():
    #     print(key, len(dataset.captions[key]))
    img1, img2, source_cap, intra_cap, inter_cap = dataset[0]
    print(np.shape(intra_cap))
    print(np.shape(inter_cap))