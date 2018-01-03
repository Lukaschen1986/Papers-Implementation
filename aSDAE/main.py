# encoding=utf-8

import sys
sys.path.append("../")

import pandas as pd
import sadae
import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error
from keras.optimizers import Adam
from dataset import MovieLens100k


if __name__ == "__main__":
    data_path="../data/"

    train_user_vec, train_user_add, train_movie_vec, train_movie_add, train_rating \
        = MovieLens100k.create_data(data_path+"ml-100k/u1.base", data_path+"ml-100k/u.user",
                                data_path+"ml-100k/u.item.n", 943, 1682)

    test_user_vec, test_user_add, test_movie_vec, test_movie_add, test_rating \
        = MovieLens100k.create_data(data_path+"ml-100k/u1.test", data_path+"ml-100k/u.user",
                                data_path+"ml-100k/u.item.n", 943, 1682)

    save = ModelCheckpoint("weights.h5", verbose=1, save_best_only=True, save_weights_only=True)
    early= EarlyStopping(patience=10, verbose=1)
    optimizer = Adam(lr=0.0001, beta_2=0.9, decay=1e-6)

    model, loss = sadae.aSDAE_reSys([train_user_vec.shape[1]], [train_user_add.shape[1]],
                              [train_movie_vec.shape[1]], [train_movie_add.shape[1]],
                              500, user_hun=[1000,800], item_hun=[700], alpha_1=0.1,
                              alpha_2=0.1, lam=0.2)

    model.compile(optimizer, loss, metrics=[keras.losses.mean_squared_error])
    model.fit(x=[train_user_vec, train_user_add, train_movie_vec, train_movie_add], y=train_rating,
              batch_size=64, epochs=100, verbose=2, callbacks=[save, early], validation_split=0.2)
    model.load_weights("weights.h5")

    y_ = model.predict(x=[test_user_vec, test_user_add, test_movie_vec, test_movie_add])

    print(mean_squared_error(test_rating, y_))

    df = pd.DataFrame()
    df["y_true"] = test_rating
    df["y_pre"] = y_
    df.to_csv("result.csv")
