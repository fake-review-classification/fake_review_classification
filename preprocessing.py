from collections import defaultdict
import re

from tqdm import tqdm

import pandas as pd
from nltk.corpus import stopwords

class preprocessing:
    def __init__(self, df, threshold=1, random_seed=42):
        self.dataframe = df
        self.threshold = threshold
        self.random_seed = 42

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
        real_review_dict = dict()
        fake_review_dict = dict()

        for i in range(df_train.shape[0]):
            # word_lst = df_train.review[i].lower().split()
            
            sentence = df_train.review[i].lower()
            sentence = re.sub('[\d]+', ' ', sentence)
            sentence = re.sub('[^A-Za-z\s]', ' ', sentence)
            # sentence = re.sub('[NUMBER]+', ' ', sentence)
            word_lst = sentence.split()

            if df_train.label[i] == 1:
                for ele in word_lst:
                    if ele in real_review_dict:
                        real_review_dict[ele] += 1
                    else:
                        real_review_dict[ele] = 1

            else:
                for ele in word_lst:
                    if ele in fake_review_dict:
                        fake_review_dict[ele] += 1
                    else:
                        fake_review_dict[ele] = 1

        return real_review_dict, fake_review_dict

    def make_del_word_lst(self, fake_review_dict, real_review_dict, threshold):
        '''단어들의 갯수를 이용해, 어떤 단어를 제거할지 찾는 함수'''
        del_word_lst = []
        sum_real_review = sum(real_review_dict.values())
        sum_fake_review = sum(fake_review_dict.values())

        for ele in fake_review_dict:
            if ele in real_review_dict:
                real_cnt = real_review_dict[ele] / sum_real_review
                fake_cnt = fake_review_dict[ele] / sum_fake_review

                if real_cnt < fake_cnt:
                    real_cnt, fake_cnt = (fake_cnt, real_cnt)
        
                if (fake_cnt / real_cnt) >= threshold:
                    del_word_lst.append(ele)
        
        print(del_word_lst)
        return del_word_lst

    def make_review_lst(self, df, del_word_lst):
        '''특수문자, 숫자를 제거한 뒤 위에서 찾은 단어들 삭제하는 함수'''
        new_review_lst = []
        for sentence in tqdm(df.review):
            sentence = sentence.lower()

            sentence = re.sub('[\d]+', ' ', sentence)
            sentence = re.sub('[^A-Za-z\s]', '', sentence)
            # sentence = re.sub('[<NUM>]+', '<NUM>', sentence) # if you don't want to delete the number
            sentence = re.sub('[<NUM>]+', '', sentence) # if you want to delete the number
            new_review_lst.append(''.join(map(lambda x: x + ' ' if x not in del_word_lst else '', sentence.split())))

        return new_review_lst

    def preprocessing_all(self, kold=0):
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

        new_label_lst = []

        for ele in self.dataframe.label:
            if ele == -1:
                new_label_lst.append(0)
            else:
                new_label_lst.append(1)

        self.dataframe['label'] = new_label_lst

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
        
        real_review_dict, fake_review_dict = self.make_word_dict(df_train)
        
        stop_words = set(stopwords.words('english'))
        del_word_lst = self.make_del_word_lst(fake_review_dict, real_review_dict, threshold=self.threshold)
           
        del_word_lst = list(stop_words | set(del_word_lst))
        print(f'len(del_word_lst): {len(del_word_lst)}')

        train_review_lst = self.make_review_lst(df_train, del_word_lst)
        val_review_lst = self.make_review_lst(df_val, del_word_lst)
        test_review_lst = self.make_review_lst(df_test, del_word_lst)

        df_train['review'] = train_review_lst
        df_val['review'] = val_review_lst
        df_test['review'] = test_review_lst
        
        return df_train, df_val, df_test, (real_review_dict, fake_review_dict)