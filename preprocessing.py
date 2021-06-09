import random
import numpy as np
import os
import json
import re
from collections import defaultdict
from tqdm import tqdm

import pandas as pd
from nltk.corpus import stopwords


class preprocessing:
    def __init__(self, df, random_seed=42):
        self.dataframe = df
        self.random_seed = 42

        random.seed(self.random_seed)
        np.random.seed(self.random_seed)

    def make_id_dict(self):
        '''한 유저가 리뷰를 몇개를 작성했는지, 한 상품에 대해서 얼마나 많은 리뷰가 작성됐는지 카운트 하는 코드 
        (현재 모델에서는 사용하지는 않는 데이터, 추후에 어떤 식으로 사용할지 생각해보자)'''
        user_id_dict = defaultdict(int)
        prod_id_dict = defaultdict(int)

        for idx in range(self.dataframe.shape[0]):
            user_id_dict[self.dataframe.user_id[idx]] += 1

            prod_id_dict[self.dataframe.prod_id[idx]] += 1
            
        return user_id_dict, prod_id_dict

    def make_word_dict(self, df_train):
        '''문장 전처리를 수행한 뒤, 데이터에 속한 단어들의 갯수 카운팅하는 함수'''
        self.real_review_dict = dict()
        self.fake_review_dict = dict()
        self.all_word_set = set()
        
        for i in range(df_train.shape[0]):
            # word_lst = df_train.review[i].lower().split()
            
            sentence = df_train.review[i].lower()
            sentence = re.sub('[\d]+', ' ', sentence)
            sentence = re.sub('[^A-Za-z\s]', ' ', sentence)
            # sentence = re.sub('[NUMBER]+', ' ', sentence)
            word_lst = sentence.split()

            if df_train.label[i] == 1:
                for ele in word_lst:
                    if ele in self.real_review_dict:
                        self.real_review_dict[ele] += 1
                    else:
                        self.real_review_dict[ele] = 1
                    
                    self.all_word_set.add(ele)
                    
            else:
                for ele in word_lst:
                    if ele in self.fake_review_dict:
                        self.fake_review_dict[ele] += 1
                    else:
                        self.fake_review_dict[ele] = 1
                        
                    self.all_word_set.add(ele)
            
#         print('save')
#         with open('all_word.json', 'w', encoding='utf-8') as make_file:
#             json.dump({'word': list(self.all_word_set)}, make_file, ensure_ascii=False, indent='\t')
#         print('done')
                

    def make_del_word_set(self, ratio_threshold):
        '''단어들의 분포 비율을 이용해, 어떤 단어를 제거할지 찾는 함수'''
        del_word_set = set()
        
        if ratio_threshold >= 1:
            return del_word_set
        
        sum_real_review = sum(self.real_review_dict.values())
        sum_fake_review = sum(self.fake_review_dict.values())

        for ele in self.fake_review_dict:
            if ele in self.real_review_dict:
                real_cnt = self.real_review_dict[ele] / sum_real_review
                fake_cnt = self.fake_review_dict[ele] / sum_fake_review

                if real_cnt < fake_cnt:
                    real_cnt, fake_cnt = (fake_cnt, real_cnt)
        
                if (fake_cnt / real_cnt) >= ratio_threshold:
                    del_word_set.add(ele)
        
        return del_word_set

    def make_review_lst(self, df, del_word_list):
        '''특수문자, 숫자를 제거한 뒤 위에서 찾은 단어들 삭제하는 함수'''
        new_review_lst = []
        for sentence in tqdm(df.review):
            sentence = sentence.lower()

            sentence = re.sub('[\d]+', ' ', sentence)
            sentence = re.sub('[^A-Za-z\s]', '', sentence)
            # sentence = re.sub('[<NUM>]+', '<NUM>', sentence) # if you don't want to delete the number
            sentence = re.sub('[<NUM>]+', '', sentence) # if you want to delete the number
            new_review_lst.append(''.join(map(lambda x: x + ' ' if x not in del_word_list else '', sentence.split())))

        return new_review_lst
    
    @staticmethod
    def dist_del_words(word_list, dist_type=None, dist_threshold=1, folder_path='./json_folder'):
        '''dist_threshold와 단어의 거리를 비교해 추가된 지울 단어 집합을 반환하는 함수'''
        assert dist_type in ('cosine_similarity', 'c', 'euclidean_distance', 'e', None), 'wrong dist_type'
        
        add_del_word_set = set()
        
        if dist_type == 'cosine_similarity' or dist_type == 'c':
            idx2 = 0
        elif dist_type == 'euclidean_distance' or dist_type == 'e':
            idx2 = 1
        else:
            return set()

        for word in tqdm(word_list):

            file = word + '.json'

            with open(os.path.join(folder_path, file), 'r') as f:
                json_data = json.load(f)[word]

            word_dist_list = list(json_data.items())
            
            if idx2 == 0:
                word_dist_list.sort(reverse=True, key=lambda x: x[1][idx2])
            else:
                word_dist_list.sort(key=lambda x: x[1][idx2])
                
            for dist_info in word_dist_list:
                if idx2 == 0:    
                    if dist_info[1][idx2] < dist_threshold:
                        break
            
                else:
                    if dist_info[1][idx2] > dist_threshold:
                        break
                
                add_del_word_set.add(dist_info[0])

        return add_del_word_set
    
    def check_around_words(self, word_list, around_threshold, dist_type=None, dist_threshold=1, folder_path='./json_folder', soft=True):
        '''단어의 주변 단어의 분포를 체크해, 해당 단어를 제거할지 제거하지 않을지 결정해 반환하는 함수'''
        assert dist_type in ('cosine_similarity', 'c', 'euclidean_distance', 'e', None), 'wrong dist_type'
        
        del_word_set = set()
        sum_real_review = sum(self.real_review_dict.values())
        sum_fake_review = sum(self.fake_review_dict.values())
        
        if dist_type == 'cosine_similarity' or dist_type == 'c':
            idx2 = 0
        elif dist_type == 'euclidean_distance' or dist_type == 'e':
            idx2 = 1
        
        file_list = os.listdir(folder_path)

        for word in tqdm(word_list):
            
            file = word + '.json'
            
            if file not in file_list:
                continue

            with open(os.path.join(folder_path, file), 'r') as f:
                json_data = json.load(f)[word]
            
            word_dist_list = list(json_data.items())
            real_sum = 0
            fake_sum = 0
            real_cnt = 0
            fake_cnt = 0
            
            if idx2 == 0:
                word_dist_list.sort(reverse=True, key=lambda x: x[1][idx2])
            else:
                word_dist_list.sort(key=lambda x: x[1][idx2])
                
            for dist_info in word_dist_list:
                if idx2 == 0:
                    if dist_info[1][idx2] < dist_threshold:
                        break
                else:
                    if dist_info[1][idx2] > dist_threshold:
                        break

                real_value = self.real_review_dict.get(dist_info[0], 0) / sum_real_review
                fake_value = self.fake_review_dict.get(dist_info[0], 0) / sum_fake_review

                real_sum += real_value
                fake_sum += fake_value

                if real_value > fake_value:
                    real_cnt += 1
                else:
                    fake_cnt += 1
            if soft:
                diff = abs(real_sum - fake_sum)
            else:
                diff = abs(real_cnt - fake_cnt)
                
            if diff < around_threshold:
                del_word_set.add(word)
        
        return del_word_set
                    
    
    def preprocessing_all(self, ratio_threshold=1, preprocessing_function=(None, None), dist_type=None, dist_threshold=1):
        ''''''
        print('make id dictionary and count id frequency of id ...')
        user_id_dict, prod_id_dict = self.make_id_dict()

        user_id_count = []
        prod_id_count = []

        for idx in tqdm(range(self.dataframe.shape[0])):
            user_id_count.append(user_id_dict[self.dataframe.user_id[idx]])
            prod_id_count.append(prod_id_dict[self.dataframe.prod_id[idx]])

        review_lst = []
        
        for review in self.dataframe['review']:
            if len(review) > 512:
                review_lst.append(review[:512])
                continue
            review_lst.append(review)

        self.dataframe['review'] = review_lst
        del review_lst

        new_label_lst = []

        for ele in self.dataframe.label:
            if ele == -1:
                new_label_lst.append(0)
            else:
                new_label_lst.append(1)

        self.dataframe['label'] = new_label_lst
        del new_label_lst

        self.dataframe = self.dataframe.sample(frac=1, random_state=self.random_seed)
        self.dataframe = self.dataframe.reset_index(drop=True)
        print('label is changed!!! (-1, 1) => (0, 1)', end='\n\n')

        self.dataframe = self.dataframe.drop(['user_id', 'prod_id', 'date'], axis=1)

        is_real = self.dataframe['label'] == 1
        real_data = self.dataframe[is_real]

        is_fake = self.dataframe['label'] == 0
        fake_data = self.dataframe[is_fake]

        assert self.dataframe.shape[0] == real_data.shape[0] + fake_data.shape[0]

        print(f'length of real review : {len(real_data)}, length of fake review : {len(fake_data)}', end='\n\n')

        # need change here to use cross validation
        print('train val test split')
        df_train_real = real_data[:10000].reset_index(drop=True)
        df_val_real = real_data[10000:11000].reset_index(drop=True)
        df_test_real = real_data[11000:13000].reset_index(drop=True)

        df_train_fake = fake_data[:10000].reset_index(drop=True)
        df_val_fake = fake_data[10000:11000].reset_index(drop=True)
        df_test_fake = fake_data[11000:13000].reset_index(drop=True)

        df_train = pd.concat([df_train_real, df_train_fake])
        df_val = pd.concat([df_val_real, df_val_fake])
        df_test = pd.concat([df_test_real, df_test_fake])

        df_train = df_train.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        df_val = df_val.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        df_test = df_test.sample(frac=1, random_state=self.random_seed).reset_index(drop=True)
        
        self.make_word_dict(df_train)
        
        stop_words = set(stopwords.words('english'))
        if preprocessing_function[0] != 'check_around_words':
            del_word_set = self.make_del_word_set(ratio_threshold=ratio_threshold)
        
        if preprocessing_function[0] == 'check_around_words':
            print('check around')
            all_word_list = list(self.all_word_set)
            del_word_set = self.check_around_words(all_word_list, around_threshold=ratio_threshold, dist_type=dist_type, dist_threshold=dist_threshold, soft=preprocessing_function[1])
        
        if preprocessing_function[0] == 'dist_del_words':
            print('checking word distance...')
            dist_word_set = self.dist_del_words(list(del_word_set), dist_type=dist_type, dist_threshold=dist_threshold)
            
            _ = del_word_set
            del_word_set = (del_word_set | dist_word_set)
            assert del_word_set != _
            del _

        del_word_list = list(stop_words | del_word_set)

        print(f'len(del_word_list): {len(del_word_list)}')

        train_review_list = self.make_review_lst(df_train, del_word_list)
        val_review_list = self.make_review_lst(df_val, del_word_list)
        test_review_list = self.make_review_lst(df_test, del_word_list)

        df_train['review'] = train_review_list
        df_val['review'] = val_review_list
        df_test['review'] = test_review_list

        del train_review_list
        del val_review_list
        del test_review_list     

        return df_train, df_val, df_test
