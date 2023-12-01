# Note: The following confusion matrix code is a remix of Scikit-Learn's
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
# import tensorflow as tf
import pandas as pd
import zipfile

# Our function needs a different name to sklearn's plot_confusion_matrix
def make_confusion_matrix(y_true, y_pred,
                           classes=None, figsize=(10, 10), text_size=15, norm=False, savefig=False):
  """Makes a labelled confusion matrix comparing predictions and ground truth labels.

  If classes is passed, confusion matrix will be labelled, if not, integer class values
  will be used.

  Args:
    y_true: Array of truth labels (must be same shape as y_pred).
    y_pred: Array of predicted labels (must be same shape as y_true).
    classes: Array of class labels (e.g. string form). If `None`, integer labels are used.
    figsize: Size of output figure (default=(10, 10)).
    text_size: Size of output figure text (default=15).
    norm: normalize values or not (default=False).
    savefig: save confusion matrix to file (default=False).

  Returns:
    A labelled confusion matrix plot comparing y_true and y_pred.

  Example usage:
    make_confusion_matrix(y_true=test_labels, # ground truth test labels
                          y_pred=y_preds, # predicted labels
                          classes=class_names, # array of class label names
                          figsize=(15, 15),
                          text_size=10)
  """
  # Create the confustion matrix
  cm = confusion_matrix(y_true, y_pred)
  cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # normalize it
  n_classes = cm.shape[0] # find the number of classes we're dealing with

  # Plot the figure and make it pretty
  fig, ax = plt.subplots(figsize=figsize)
  cax = ax.matshow(cm, cmap=plt.cm.Blues) # colors will represent how 'correct' a class is, darker == better
  fig.colorbar(cax)

  # Are there a list of classes?
  if classes:
    labels = classes
  else:
    labels = np.arange(cm.shape[0])

  # Label the axes
  ax.set(title="Confusion Matrix",
         xlabel="Predicted label",
         ylabel="True label",
         xticks=np.arange(n_classes), # create enough axis slots for each class
         yticks=np.arange(n_classes),
         xticklabels=labels, # axes will labeled with class names (if they exist) or ints
         yticklabels=labels)

  # Make x-axis labels appear on bottom
  ax.xaxis.set_label_position("bottom")
  ax.xaxis.tick_bottom()

  ### Added: Rotate xticks for readability & increase font size (required due to such a large confusion matrix)
  plt.xticks(rotation=70, fontsize=text_size)
  plt.yticks(fontsize=text_size)

  # Set the threshold for different colors
  threshold = (cm.max() + cm.min()) / 2.

  # Plot the text on each cell
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    if norm:
      plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j]*100:.1f}%)",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)
    else:
      plt.text(j, i, f"{cm[i, j]}",
              horizontalalignment="center",
              color="white" if cm[i, j] > threshold else "black",
              size=text_size)

  # Save the figure to the current working directory
  if savefig:
    fig.savefig("confusion_matrix.png")



def binary_plot_loss_history_confiusion_matrix(model,pl_history="", pl_X_test="", pl_y_test="",tensorflow_dataset=False,
                                               classes=None,lost_fig_size=(4, 4), figsize=(10, 10),
                                               text_size=15, norm=True, savefig=False,
                                               save_model=False,save_path=""):

  """ Make skelearn report with loass and accuracy plot with learning rate if preset
  and confusion matrix and loss history also showing higest validation accuracy against learning rate also
  option to save model and confusion matrix to file, if tensorflow dataset is used yhen X_test assume that
   batch size is X_test position

  Arug:
    model: model
    pl_history: history of model
    pl_X_test: test data input
    pl_y_test: test data output
    tensorflow_dataset:  default = False
    classes: classes if avilable else None
    lost_fig_size: size of lossand accuracy plot
    figsize: size of confusion matrix
    text_size: size of text in confusion matrix
    norm: normalize confusion matrix with % result in confusion matrix
    savefig: save confusion matrix to file (default=False)
    save_model: save model( default=False)
    save_path: path to save model (default="")
    """


  # if dataset is tensorflow dataset
  if tensorflow_dataset:
    # Extract images and labels from the TensorFlow dataset
    image_data = []
    label_data = []

    for batch in pl_X_test.unbatch().as_numpy_iterator():
        images, labels = batch
        image_data.extend(images)
        label_data.extend([labels] * len(images))  # Convert single label to a list

    # Convert the lists to NumPy arrays
    pl_X_test = np.array(image_data)
    pl_y_test = np.array(label_data)

  # make report with skelearn metrics

  from sklearn.metrics import classification_report

  # made pl_y_pred from pl_y_pred > 0.5

  pl_y_pred = model.predict(pl_X_test)

  # evluate model:
  print('Evluate model:')
  evaluate_data =  model.evaluate(pl_X_test, pl_y_test)
  print()

  # Save model
  if save_model:
    model.save(f"{save_path}_{round((evaluate_data[1]),2)}")

    print(f"Model saved to {save_path}_{round((evaluate_data[1]),2)}")
    print()

  # try if learning rate
  try :
    if pl_history.history['lr']:
      learing_data = pd.DataFrame(pl_history.history)
      # Find 5 highest val_accuracy and show them in plot
      learing_data['epoch'] = learing_data.index + 1
      learing_data = learing_data.sort_values('val_accuracy', ascending=False)
      learing_data = learing_data.head(5)

      # print(learing_data)
      print(learing_data[["lr",'val_accuracy',"val_loss"]])
      print()

  except:
    pass


  if classes:
    print(classification_report(pl_y_test, pl_y_pred > 0.5,target_names=classes))
  else:
    print(classification_report(pl_y_test, pl_y_pred > 0.5))

  print()

  # plot loss history

  plt.figure(figsize=lost_fig_size)
  plt.subplot(2, 1, 1)
  plt.plot(pl_history.history['loss'])
  plt.plot(pl_history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')


  plt.subplot(2, 1, 2)
  plt.plot(pl_history.history['accuracy'])
  plt.plot(pl_history.history['val_accuracy'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')

  plt.show()

  # make confusion matrix
  make_confusion_matrix(pl_y_test, pl_y_pred > 0.5, classes=classes, figsize=figsize, text_size=text_size, norm=norm, savefig=savefig)


# function to plot loss and
def plot_loss_and_learning_rate(pl_history,pl_model=None,pl_X_test=None, pl_y_test=None,save_model=False,save_path=""):

  """ plot loss and learning rate if preset for not categorical model
      able to save model in specific path give evaluated data to and show 5 best learning rate

    Arug:
    pl_history: history of model
    save_model: save model( default=False)
    save_path: path to save model (default="")
    model: model (default=None)
    pl_X_test: test data input (default=None)
    pl_y_test: test data output (default=None)
    """

  # plot loss history
  plt.figure(figsize=(10, 10))

  plt.plot(pl_history.history['loss'])
  plt.plot(pl_history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'val'], loc='upper left')

  # try if learning rate
  try :
    if pl_history.history['lr']:
      learing_data = pd.DataFrame(pl_history.history)
      # Find 5 highest val_accuracy and show them in plot
      learing_data['epoch'] = learing_data.index + 1
      learing_data = learing_data.sort_values('val_accuracy', ascending=False)
      learing_data = learing_data.head(5)

      # print(learing_data)
      print(learing_data[["lr","val_loss"]])
      print()

  except:
    pass

  if pl_model and pl_X_test and pl_y_test:
    evaluate_data =  pl_model.evaluate(pl_X_test, pl_y_test)

  # Save model
  if save_model and pl_model:
    pl_model.save(f"{save_path}_{evaluate_data[1]}")

    print(f"Model saved to {save_path}_{evaluate_data[1]}")


# uzip file
def unzip_data(zip_file,uzipath=""):
  """ Unzip a file. and if unzip path is given then unzip to that path othwerwise unzip to current directory """
  zip_ref = zipfile.ZipFile(zip_file, 'r')
  if uzipath != "":
    zip_ref.extractall(uzipath)
    print("unzipped to ",uzipath)
  else:
    zip_ref.extractall()
    print("unzipped to current directory")

  zip_ref.close()





""" All code below this is main code made by me  """


from pandas.core import series
# make data from Moon_jellyfish folder
from os import listdir
from os.path import isfile, join
import pandas as pd
import numpy as np
import tensorflow as tf


def prpeare_data_from_folder(main_path,main_data_return=True,numpy_return=False,commits=""):
  """ Collect image and label from folder then create dataframe for X and y

  Args:
    main_path: path to folder in which image and label are prsent
    main_data_return: return dataframe with image and label
    numpy_return: return numpy array with seperate image and label
    commits: commits in which data is collected (default="") can be traning or testing or validation
  """

  classification = []
  for i in listdir(main_path):
    # drop csv file extension
    # print(i)
    classification.append(i)

  # loop through each folder
  image_data = []
  label_data = []

  for image_path in classification:
    for image in listdir(main_path+"/"+image_path):
      image_data.append(main_path+"/"+image_path+"/"+image)
      label_data.append(image_path)

  df = pd.DataFrame({

    'image': image_data,
    'label' : label_data

})

  # print dataframe quantity in
  print(f" For {commits} Total images: {len(df)} and Total labels: {len(classification)}")

  # if numpy return True then return X and y in numpy array
  if numpy_return:
    return df['image'].to_numpy(), df['label'].to_numpy()
  # return data dataframe combine
  elif main_data_return:
    return df
  # return image and label in dataframe
  else:
    return df['image'], df['label']



# preperforning data into tensor
def prepare_data_to_tensor(x, y=None, batch_size=32, valid_data=False, test_data=False,
                        IMG_SIZE=224,rescale=True):

  def process_image(image_path):


    """
    Takes an image from file path convert into 3 colour channel tensor then resize it and return .
    """
    # Read in image file
    image = tf.io.read_file(image_path)
    # Turn the jpeg image into numerical Tensor with 3 colour channels (Red, Green, Blue)
    image = tf.image.decode_jpeg(image, channels=3)
    # Convert the colour channel values from 0-225 values to 0-1 values
    if rescale:
      image = tf.image.convert_image_dtype(image, tf.float32)
    # Resize the image to our desired size (224, 244)
    image = tf.image.resize(image, size=[IMG_SIZE, IMG_SIZE])
    return image


  # Create a simple function to return a tuple (image, label)
  def get_image_label(image_path, label):
    """
    Takes an image file path name and the associated label,
    processes the image and returns a tuple of (image, label).
    """
    image = process_image(image_path)
    return image, label

  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data or test data.
  .
  """
  tf.random.set_seed(42)

  # Autotune for the number of available CPU threads
  AUTOTUNE = tf.data.AUTOTUNE

  # If the data is a test dataset
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label)

    # Batch and prefetch the data
    batch_size = len(x)  # Set batch size to the number of test samples
    data_batch = data_batch.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    # Extract features and labels
    test_images, test_labels = next(iter(data_batch))

    return np.array(test_images), np.array(test_labels)
        
   

  # If the data if a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    return data_batch

  else:
    # If the data is a training dataset, we shuffle it
    print("Creating training data batches...")
    # Turn filepaths and labels into Tensors
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                              tf.constant(y))) # labels

    # Shuffling pathnames and labels before mapping image processor function is faster than shuffling images
    data = data.shuffle(buffer_size=len(x))

    # Create (image, label) tuples (this also turns the image path into a preprocessed image)
    data = data.map(get_image_label)

    # Turn the data into batches
    data_batch = data.batch(batch_size ).prefetch(buffer_size=AUTOTUNE)
  return data_batch


def tensorflow_batch_to_evalauate(tensor_data):
    """
    Convert batch tensorflow tensor data to numpy array for evaluate

    Args:
        tensor_data (tf.data): tensorflow data

    Returns:
        np.array: numpy array of image and label    
    """
    # Extract the features and labels from the test dataset
    test_images, test_labels = zip(*tensor_data)

    # Convert the lists to NumPy arrays
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)

    return test_images, test_labels


import pandas as pd

def transform_dataframe_into_train(data, target, balance_df=True, split=0.2):
    """
    transform dataframe into train and test with split proportion with class balance

    Arg:
      df: dataframe
      target: target column
      balance_df: if True, balance dataframe based on target column values and sample with replacement based on min value in target
      split: proportion of train data
    Return:
      train_x: train features
      train_y: train target
      test_x: test features
      test_y: test target
      
    """
    df = data.copy()

    # Shuffle the dataframe
    df = df.sample(frac=1).reset_index(drop=True)

    # print how many tota sample are in data then also print how many samples in each class
    print("Total samples",df.shape[0])
    print(df[target].value_counts())

    
    if balance_df:
        # Balance dataframe based on target column values and sample with replacement based on min value in target
        df = df.groupby(target).apply(lambda x: x.sample(df[target].value_counts().min(), replace=True)).reset_index(drop=True)
        
        # print how many samples in each class
        print("Balance set",df[target].value_counts())

    # Split dataframe into train and valid
    train_size = 1 - split
    train_dfs, valid_dfs = [], []

    for _, group_df in df.groupby(target):
        group_size = len(group_df)
        group_train_size = int(group_size * train_size)

        # Split each group into train and valid proportionally
        train_df = group_df.head(group_train_size)
        valid_df = group_df.tail(group_size - group_train_size)

        train_dfs.append(train_df)
        valid_dfs.append(valid_df)

    # Concatenate the dataframes for train and valid sets
    df_train = pd.concat(train_dfs).sample(frac=1).reset_index(drop=True)
    df_valid = pd.concat(valid_dfs).sample(frac=1).reset_index(drop=True)

    # shaufle df
    df_train = shuffle(df_train, random_state=42)
    df_valid = shuffle(df_valid, random_state=42)

    # split into train_x and train_y
    train_x = df_train.drop(target, axis=1)
    train_y = df_train[target]

    # split into valid_x and valid_y
    valid_x = df_valid.drop(target, axis=1)
    valid_y = df_valid[target]

    # print how many samples in each set
    print("Return train \n",df_train[target].value_counts())
    print("Return valid \n",df_valid[target].value_counts())

    return train_x, train_y, valid_x, valid_y



def make_image_database_from_folder(train_path="", valid_path="", test_path="",
                                    chart_figure=(4, 8), before_process=True,
                                    numpy_return=False, main_data_return=False, BATCH_SIZE=32,
                                    IMG_SIZE=224, rescale=True, train_balance=False, valid_balance=False,
                                    train_percent=1.0, valid_percent=1.0,
                                    true_false_output=False):
    """
        Convert image and label from folder then create dataframe for X and y and if before_process True then
        return dataset else return X and y

        Args:
            train_path: path of train folder
            valid_path: path of valid folder
            test_path: path of test folder
            chart_figure: figure size for chart showing dataset
            before_process: if True then return dataset else return X and y
            numpy_return: if True then return X and y in numpy array
            main_data_return: if True then return dataframe
            BATCH_SIZE: batch size
            IMG_SIZE: image size
            rescale: if True then rescale image
            train_balance: if True then balance train data
            valid_balance: if True then balance valid data
            train_percent: percentage of train data to reduce if less than 1
            valid_percent: percentage of valid data to reduce if less than 1
            tue_flase_output: if True then return True or False number array

        return:
            if before_process is True then return dataset else return X and y
            if main_data_return is True then return dataframe
            if numpy_return is True then return X and y in numpy array

        example:
               if before_process is True:
                return classfication data , train data tensor, valid data tensor ,and in Numpy array of Image and label


    """
    import random
    # set random seed
    random.seed(42)

    # Initialize separate classification lists for training and validation
    train_classification = []
    valid_classification = []

    for i in listdir(train_path):
        train_classification.append(i)

    for i in listdir(valid_path):
        valid_classification.append(i)

    if len(train_classification) == 0 or len(valid_classification) == 0:
        print("No files found in train or valid folders")
        return None

    # train path
    if train_path:
        print("train original data: ")
        for s in train_classification:
            path_found = len(listdir(f"{train_path}/{s}"))
            print(f"{s} : {path_found}")
        print()

        # Empty lists
        image_data_train = []
        label_data_train = []

        if train_balance:
        # loop through each folder quantity and find out min value quantity
            min_balance = []
            for s in train_classification:
                path_found = len(listdir(f"{train_path}/{s}"))
                min_balance.append(path_found)

            # find out minimum value in minimum balance
            min_balance_value = min(min_balance)
            print("min_balance: ", min_balance_value)

            for train_image_path in train_classification:
                sfit_path = listdir(train_path + "/" + train_image_path)

                # loop through each randomly shuffled sfit_path
                random_shift = random.sample(sfit_path, min_balance_value)

                 # if train_percent < 1.0 then *train_percent round to integer
                if train_percent < 1.0:
                # *train_percent round to integer if train_percent < 1.0
                    min_balance_adjusted = int(min_balance_value * train_percent)

                    # loop through each randomly shuffled sfit_path only select min_balance_adjusted
                    for i in range(min_balance_adjusted):
                        image_data_train.append(train_path + "/" + train_image_path + "/" + random_shift[i])
                        label_data_train.append(train_image_path)
                else:
                    # loop through each randomly shuffled sfit_path only select min_balance_value
                    for i in range(min_balance_value):
                        image_data_train.append(train_path + "/" + train_image_path + "/" + random_shift[i])
                        label_data_train.append(train_image_path)

        else:
            for train_image_path in train_classification:
                # if train_percent < 1.0 then *train_percent round to integer
                if train_percent < 1.0:
                    min_balance = int(len(listdir(train_path + "/" + train_image_path)) * train_percent)

                    # loop through each randomly shuffled sfit_path only select min_balance but first shuffle sfit_path
                    train_shift_path = random.sample(listdir(train_path + "/" + train_image_path), min_balance)

                    for i in range(min_balance):
                        image_data_train.append(train_path + "/" + train_image_path + "/" + listdir(train_path + "/" + train_image_path)[i])
                        label_data_train.append(train_image_path)
                else:
                    for image in listdir(train_path + "/" + train_image_path):
                        image_data_train.append(train_path + "/" + train_image_path + "/" + image)
                        label_data_train.append(train_image_path)

        train_df = pd.DataFrame({
            'image': image_data_train,
            'label': label_data_train
        })

        # How many images are there of each breed?
        train_df["label"].value_counts().plot.bar(figsize=chart_figure, title=" Train Dataset")
        plt.show()

        # print dataframe quantity in
        print(f" For Train Total images: {len(train_df)} and Total labels: {len(train_classification)}")

        # print total quantity of each label
        # print(train_df["label"].value_counts())

    # for valid path
    if valid_path:
        print("valid original data: ")
        for s in valid_classification:
            path_found = len(listdir(f"{valid_path}/{s}"))
            print(f"{s} : {path_found}")
        print()

        # loop through each folder
        image_data_valid = []
        label_data_valid = []

        if valid_balance:
            min_balance = []
            for s in valid_classification:
               path_found = len(listdir(f"{valid_path}/{s}"))
               min_balance.append(path_found)

            # find out minimum value in minimum balance
            min_balance_value = min(min_balance)

            for valid_image_path in valid_classification:
                sfit_path = listdir(valid_path + "/" + valid_image_path)

                # shuffle sfit_path
                random_shift = random.sample(sfit_path, min_balance_value)
                # print("sfit_path: ",random_shift)

                # if valid_percent < 1.0 then *valid_percent round to integer
                if valid_percent < 1.0:
                # *valid_percent round to integer if valid_percent < 1.0
                    min_balance_adjusted = int(min_balance_value * valid_percent)

                    # loop through each randomly shuffled sfit_path only select min_balance_adjusted
                    for i in range(min_balance_adjusted):
                        # print("valid_image_path: ", random_shift[i])
                        image_data_valid.append(valid_path + "/" + valid_image_path + "/" + random_shift[i])
                        label_data_valid.append(valid_image_path)
                else:
                    # loop through each randomly shuffled sfit_path only select min_balance_value
                    for i in range(min_balance_value):
                       # print("valid_image_path: ", random_shift[i])
                       image_data_valid.append(valid_path + "/" + valid_image_path + "/" + random_shift[i])
                       label_data_valid.append(valid_image_path)

        else:
            for valid_image_path in valid_classification:

                if valid_percent < 1.0:
                    min_balance = int(len(listdir(valid_path + "/" + valid_image_path)) * valid_percent)

                    # loop through each randomly shuffled sfit_path only select min_balance but first shuffle sfit_path
                    valid_shift_path = random.sample(listdir(valid_path + "/" + valid_image_path), min_balance)

                    # loop through each randomly shuffled sfit_path minimum balance quantity
                    for i in range(min_balance):
                        image_data_valid.append(valid_path + "/" + valid_image_path + "/" + listdir(valid_path + "/" + valid_image_path)[i])
                        label_data_valid.append(valid_image_path)
                else:
                    # if valid_percent == 1.0 then select all
                    for image in listdir(valid_path + "/" + valid_image_path):
                        image_data_valid.append(valid_path + "/" + valid_image_path + "/" + image)
                        label_data_valid.append(valid_image_path)

        # create dataframe for valid
        valid_df = pd.DataFrame({
            'image': image_data_valid,
            'label': label_data_valid
        })

        # How many images are there of each breed?
        valid_df["label"].value_counts().plot.bar(figsize=chart_figure, title=" Valid Dataset")
        plt.show()

        # print
        print(f" For Valid Total images: {len(valid_df)} and Total labels: {len(valid_classification)}")

    else:
        # empty dataframe because no valid path
        valid_df = pd.DataFrame({
            'image': [],
            'label': []
        })

    # for test path
    if test_path:
        # loop through each folder
        image_data_test = []
        label_data_test = []

        for test_image_path in train_classification:
            for image in listdir(test_path + "/" + test_image_path):
                image_data_test.append(test_path + "/" + test_image_path + "/" + image)
                label_data_test.append(test_image_path)

        test_df = pd.DataFrame({
            'image': image_data_test,
            'label': label_data_test
        })

        # How many images are there of each breed?
        test_df["label"].value_counts().plot.bar(figsize=chart_figure, title="Test Dataset")
        plt.show()

        print(f" For Test Total images: {len(test_df)} and Total labels: {len(train_classification)}")

    else:
        # empty dataframe
        test_df = pd.DataFrame({
            'image': [],
            'label': []
        })

    # before process
    if before_process:
        # if numpy return True then return X and y in numpy array
        if numpy_return:
            return train_classification, train_df['image'].to_numpy(), train_df['label'].to_numpy(), valid_df['image'].to_numpy(), valid_df['label'].to_numpy(), test_df['image'].to_numpy(), test_df['label'].to_numpy()
        # return data dataframe combine
        elif main_data_return:
            return train_classification, train_df, valid_df, test_df
        # return image and label in dataframe
        else:
            return train_classification, train_df['image'], train_df['label'], valid_df['image'], valid_df['label'], test_df['image'], test_df['label']

    # not before process
    else:
        if true_false_output:
            unique = np.unique(train_classification)

            train_label = [ label_class == np.array(unique) for label_class in train_df['label'].to_numpy()]
            valid_label = [ label_class == np.array(unique) for label_class in valid_df['label'].to_numpy()]
            test_label = [ label_class == np.array(unique) for label_class in test_df['label'].to_numpy()]

        else:
           # use label hot encoding
           from sklearn.preprocessing import LabelEncoder

           label_encoder = LabelEncoder()
           # Fit the labels into a one-hot encoding so it can be used in the data
           train_classification = label_encoder.fit(np.array(train_classification))

           # Transform the labels into a one-hot encoding
           train_label = label_encoder.transform(train_df['label'].to_numpy())
           valid_label = label_encoder.transform(valid_df['label'].to_numpy())
           test_label = label_encoder.transform(test_df['label'].to_numpy())
           
        # prepare data to tensor with batch size, image size and rescale or not
        train_data_set = prepare_data_to_tensor(train_df['image'], train_label, batch_size=BATCH_SIZE, IMG_SIZE=IMG_SIZE, rescale=rescale)
        valid_data_set = prepare_data_to_tensor(valid_df['image'], valid_label, batch_size=BATCH_SIZE, IMG_SIZE=IMG_SIZE, rescale=rescale, valid_data=True)
        test_data_set = prepare_data_to_tensor(test_df['image'], test_label, batch_size=BATCH_SIZE, IMG_SIZE=IMG_SIZE, rescale=rescale, test_data=True)

        # return data with classification and transform train, valid and last it test in numpy array Image and Label
        return train_classification, train_data_set, valid_data_set, test_data_set[0], test_data_set[1]

