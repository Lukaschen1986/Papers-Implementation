# encoding=utf-8

import pandas as pd
import numpy as np


class MovieLens100k:
    @staticmethod
    def read_user_info(filename):
        user_info = pd.read_csv(filename, header=None, sep="|", usecols=[0, 1, 2, 3])
        user_info.columns = ["user_id", "age", "gender", "occupation"]
        user_info["gender"] = user_info["gender"].map(lambda x: int(x == "M"))
        user_info = pd.get_dummies(user_info, prefix="o", columns=["occupation"])
        user_info.columns = ["user_id", "age", "gender", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
                             "11", "12","13", "14", "15", "16", "17", "18", "19", "20", "21"]
        return user_info


    @staticmethod
    def read_movie_info(filename):
        item_info = pd.read_csv(filename, header=None, sep="|",
                                usecols=[0, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23])
        item_info.columns = ["movie_id", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16",
                             "17", "18", "19", "20", "21", "22", "23"]
        return item_info


    @staticmethod
    def create_matrix(data_path, user_num=943, movie_num=1682):
        matrix = np.zeros((user_num, movie_num), dtype=np.float32)
        for line in open(data_path):
            line = line.strip().split("\t")
            user_id = int(line[0])
            movie_id = int(line[1])
            rating = int(line[2])
            matrix[user_id - 1][movie_id - 1] = rating
        return matrix


    @staticmethod
    def create_data(data_path, user_file, item_file, user_num=943, movie_num=1682):
        user_info = MovieLens100k.read_user_info(user_file)
        df_user = pd.read_csv(data_path, header=None, sep="\t", usecols=[0, 1, 2])
        df_user.columns = ["user_id", "movie_id", "rating"]
        df_user = df_user[["user_id", "rating"]].merge(user_info, on="user_id", how="left")

        movie_info = MovieLens100k.read_movie_info(item_file)
        df_item = pd.read_csv(data_path, header=None, sep="\t", usecols=[0, 1, 2])
        df_item.columns = ["user_id", "movie_id", "rating"]
        df_item = df_item[["movie_id"]].merge(movie_info, on="movie_id", how="left")

        matrix = MovieLens100k.create_matrix(data_path, user_num, movie_num)
        user_vec = []
        for user_id in df_user["user_id"].values:
            user_vec.append(matrix[user_id-1])
        movie_vec = []
        for movie_id in df_item["movie_id"].values:
            movie_vec.append(matrix[:,movie_id-1])

        rating = df_user["rating"].values
        user_add = df_user.values[:, 2:]
        movie_add = df_item.values[:, 1:]

        return np.array(user_vec), user_add, np.array(movie_vec), movie_add, rating