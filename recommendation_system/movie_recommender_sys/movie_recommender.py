import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import pickle
import csv
from surprise import Dataset, Reader, accuracy, SVD
from surprise.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity



class content_based_rcmd:
    '''
    基于内容的推荐
    电影的内容主要为tags和genres，我们合并两者，形成新的label，作为电影的内容。
    然后再进行向量化（使用TD-IDF），最后计算（余弦）相似度
    '''
    def __init__(self, movies_file, ratings_file, tags_file):
        self.movies_file = movies_file
        self.ratings_file = ratings_file
        self.tags_file = tags_file
        self.mat_path = "./storage/content_based.pkl"
        self.__pre_process()
    def __pre_process(self):
        movies = pd.read_csv(self.movies_file)
        movies['genres'] = movies['genres'].apply(lambda x: x.replace('|', ' '))
        tags = pd.read_csv(self.tags_file)
        tags.drop(['timestamp'], 1, inplace=True)
        mixed = pd.merge(movies, tags, on='movieId', how='left')
        mixed.fillna("", inplace=True)
        mixed = pd.DataFrame(mixed.groupby('movieId')['tag'].apply(lambda x: ' '.join(x)))
        movies_cont = pd.merge(movies, mixed, on='movieId', how='left')
        movies_cont['content'] = movies_cont[['tag', 'genres']].apply(lambda x: ' '.join(x), axis=1)
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(movies_cont['content'])

        tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), index=movies_cont.index.tolist())
        # LSA（ latent semantic analysis）from sklearn doc
        compress_dim = int(tfidf_df.shape[1]/6)

        svd = TruncatedSVD(n_components=compress_dim)
        latent_mat = svd.fit_transform(tfidf_df)
        cumsum = svd.explained_variance_ratio_.cumsum()
        self.__show_compression(cumsum)
        # 使用movieId来访问df
        latent_mat_df = pd.DataFrame(latent_mat, index=movies_cont['movieId'].tolist())
        file = open(self.mat_path, 'wb')
        pickle.dump(latent_mat_df, file)
        return
    def __show_compression(self, cumsum):
        plt.plot(cumsum, '.-', ms=16, color='red')
        plt.xlabel('singular value components', fontsize=12)
        plt.ylabel('cumulative percent of variance', fontsize=12)
        plt.show()
    def predict_top_n(self, movieId, n = 10):
        file = open(self.mat_path, 'rb')
        latent_mat_df = pickle.load(file)
        cont_vec = np.array(latent_mat_df.loc[movieId]).reshape(1,-1)
        cont_sim = cosine_similarity(latent_mat_df, cont_vec).resharpe(-1)
        cont_sim_df = pd.DataFrame({'cont_sim' : cont_sim}, latent_mat_df.index.tolist())
        cont_sim_df.sort_values('cont_sim', ascending=False, inplace=True)
        return cont_sim_df.head(n + 1)[1:].index.tolist()

    def get_cont_sim_mat(self):
        file = open(self.mat_path, 'rb')
        latent_mat_df = pickle.load(file)
        return latent_mat_df

class collab_filter_rcmd:
    '''
    itemCF，用户的行为变化频率慢，且电影更新的速度不会太快，因此适合用itemCF
    '''
    def __init__(self, movies_file, ratings_file, tags_file):
        self.movies_file = movies_file
        self.ratings_file = ratings_file
        self.tags_file = tags_file
        self.mat_path = "./storage/collaborative_filter.pkl"
        self.__pre_process()
    def __pre_process(self):
        movies = pd.read_csv(self.movies_file)
        ratings = pd.read_csv(self.ratings_file)
        ratings.drop(['timestamp'], 1, inplace=True)

        ratings_merged = pd.merge(movies, ratings, on="movieId", how="left")
        ratings_mat = ratings_merged.pivot(index='movieId', columns='userId', values='rating').fillna(0)

        # 这里用户数量小于电影数量，不需要分解降维
        file = open(self.mat_path, 'wb')
        pickle.dump(ratings_mat, file)

    def predict_top_n(self, movieId, n = 10):
        '''
        推荐top n 个相似的电影的id
        :param movieId:
        :param n:
        :return:
        '''
        file = open(self.mat_path, 'rb')
        latent_mat_df = pickle.load(file)
        collab_vec = latent_mat_df.loc[movieId].reshape(1, -1)
        collab_sim = cosine_similarity(latent_mat_df,collab_vec).reshape(-1)
        collab_sim_df = pd.DataFrame({'collab_sim' : collab_sim}, latent_mat_df.index.tolist())
        collab_sim_df.sort_values('collab_sim', ascending=False, inplace=True)
        return collab_sim_df.head(n+1)[1:].index.tolist()
    def get_collab_sim_mat(self):
        file = open(self.mat_path, 'rb')
        latent_mat_df = pickle.load(file)
        return latent_mat_df


class svd_collab_filter_rcmd:
    def __init__(self, movies_file, ratings_file, tags_file):
        self.movies_file = movies_file
        self.ratings_file = ratings_file
        self.tags_file = tags_file
        self.algo_path = "./storage/model_svd.pkl"
        self.__pre_process()

    def __pre_process(self):
        movies = pd.read_csv(self.movies_file)
        ratings = pd.read_csv(self.ratings_file)
        ratings.drop(['timestamp'], 1, inplace=True)
        ratings_merged = pd.merge(movies, ratings, on="movieId", how="right") # right merge，否则有的电影没有被评价
        reader = Reader(rating_scale=(1, 5))
        data = Dataset.load_from_df(ratings_merged[['userId', 'movieId', 'rating']], reader)

        trainset, testset = train_test_split(data, test_size=0.2)

        algorithm = SVD()
        algorithm.fit(trainset)
        accuracy.rmse(algorithm.test(testset))

        # 将所有数据再次训练一次
        algorithm.fit(data.build_full_trainset())
        with open(self.algo_path, 'wb') as f:
            pickle.dump(algorithm, f, pickle.HIGHEST_PROTOCOL)
        return
    def predict_usr_rating(self, userId, n=10):
        f = open(self.algo_path, 'rb')
        algorithm = pickle.load(f)
        ratings = pd.read_csv(self.ratings_file)
        mv_list = ratings[ratings.userId == userId].movieId.tolist()
        pred_list = []
        for i in mv_list:
            predicted = algorithm.predict(userId, i)
            pred_list.append((i, predicted[3]))
        pred_df = pd.DataFrame(pred_list, columns=['movieId', 'rating'])
        pred_df.sort_values('rating', ascending=False, inplace=True)
        return pred_df.head(n)['movieId'].tolist()



class hybrid_rcmd:
    '''
    两种方法得到的相似度进行结合，选出综合最像的电影
    '''
    def __init__(self, movies_file, ratings_file, tags_file):
        self.content_sim_mat = content_based_rcmd(movies_file, ratings_file, tags_file).get_cont_sim_mat()
        self.collab_sim_mat = collab_filter_rcmd(movies_file, ratings_file, tags_file).get_collab_sim_mat()
    def predict_top_n(self, movieId, n=10):
        cont_vec = np.array(self.content_sim_mat.loc[movieId]).reshape(1, -1)
        collab_vec = np.array(self.collab_sim_mat.loc[movieId]).reshape(1, -1)
        cont_sim = cosine_similarity(self.content_sim_mat, cont_vec).reshape(-1)
        collab_sim = cosine_similarity(self.collab_sim_mat, collab_vec).reshape(-1)
        hybrid_sim = ((cont_sim + collab_sim) / 2.0)
        hybrid_sim_df = pd.DataFrame({'hybrid':hybrid_sim}, self.content_sim_mat.index.tolist())
        hybrid_sim_df.sort_values('hybrid', ascending=False, inplace=True)
        return hybrid_sim_df.head(n+1)[1:].index.tolist()



class movie_recommender:
    def __init__(self):
        self.movies_file = "./ml-latest-small/movies.csv"
        self.ratings_file = "./ml-latest-small/ratings.csv"
        self.tags_file = "./ml-latest-small/tags.csv"
        self.hybrid =  hybrid_rcmd(self.movies_file, self.ratings_file, self.tags_file)
        self.svd_model = svd_collab_filter_rcmd(self.movies_file, self.ratings_file, self.tags_file)

    def recommend(self, threshold, n=10):
        recommeded = {}
        ratings = pd.read_csv(self.ratings_file)
        uid_list = ratings.userId.unique().tolist()
        for uid in uid_list:
            print(uid)
            high_ratings = ratings[(ratings['userId'] == uid) & (ratings['rating'] >= threshold)].drop_duplicates()
            if len(high_ratings) == 0:
                recommeded[uid] = self.svd_model.predict_usr_rating(uid, n)
            else:
                high_ratings.sort_values('rating')
                high_ratings_list = high_ratings.head(10)['movieId'].tolist()
                rcmd_list = [self.hybrid.predict_top_n(i,1)[0] for i in high_ratings_list]
                rcmd_list = list(set(rcmd_list))
                recommeded[uid] = rcmd_list
            print(recommeded[uid])
        with open("./movie.csv", "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["userId", "movieId"])
            for userId, mv_list in recommeded.items():
                for movieId in mv_list:
                    writer.writerow([userId, movieId])


recommender = movie_recommender()
recommender.recommend(4.0)