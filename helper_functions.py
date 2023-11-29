
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
                                               classes=None,lost_fig_size=(10, 10), figsize=(10, 10),
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



def process_image(image_path,IMG_SIZE=224,rescale=True):
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

# Define the batch size, 32 is a good default
BATCH_SIZE = 32

# Create a function to turn data into batches
def create_data_batches(x, y=None, batch_size=BATCH_SIZE, valid_data=False, test_data=False):
  """
  Creates batches of data out of image (x) and label (y) pairs.
  Shuffles the data if it's training data but doesn't shuffle it if it's validation data or test data.
  .
  """
  # Autotune for the number of available CPU threads
  AUTOTUNE = tf.data.AUTOTUNE

  # If the data is a test dataset
  if test_data:
    print("Creating test data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label)
    return data_batch

  # If the data if a valid dataset, we don't need to shuffle it
  elif valid_data:
    print("Creating validation data batches...")
    data = tf.data.Dataset.from_tensor_slices((tf.constant(x), # filepaths
                                               tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
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
    data_batch = data.batch(BATCH_SIZE).prefetch(buffer_size=AUTOTUNE)
  return data_batch



# make function to collect image and label data from dictonary
def make_image_database_from_folder(train_path="",valid_path="",test_path="",comments=""
,before_process=True,numpy_return=False,main_data_return=False,BATCH_SIZE=32):
  """
  Takes a folder path and returns a tuple of (train_image, train_label), (valid_image, valid_label), (test_image, test_label)

  returns
   numpy array:
    return calssfication and tuple of (train_image, train_label), (valid_image, valid_label), (test_image, test_label)

   main dataframe:
    return calssfication and pandas data frame (train_image, train_label), (valid_image, valid_label), (test_image, test_label)
   else:
    return classification and partially train_image , train_label, valid_image, valid_label, test_image, test_label

  """
  classification = []
  for i in listdir(train_path):
    # drop csv file extension
    # print(i)
    classification.append(i)

  if len(classification) == 0:
    print("No files found in train folder")
    return None


  # train path
  if train_path:
    # loop through each folder
    image_data_train = []
    label_data_train = []

    for train_image_path in classification:
      for image in listdir(train_path+"/"+train_image_path):
        image_data_train.append(train_path+"/"+train_image_path+"/"+image)
        label_data_train.append(train_path)

    train_df = pd.DataFrame({

    'image': image_data_train,
    'label' : label_data_train

    })

    # How many images are there of each breed?
    train_df["label"].value_counts().plot.bar(figsize=(20, 10)).title(" Train Dataset");

    # print dataframe quantity in
    print(f" For Train Total images: {len(train_df)} and Total labels: {len(classification)}")
    

  # for valid path
  if valid_path:
     # loop through each folder
    image_data_valid = []
    label_data_valid = []

    for valid_image_path in classification:
      try:
        for image in listdir(valid_path+"/"+valid_image_path):
          image_data_valid.append(valid_path+"/"+valid_image_path+"/"+image)
          label_data_valid.append(valid_path)

        valid_df = pd.DataFrame({
          'image' : image_data_valid,
         'label' : label_data_valid
           })
        
        # How many images are there of each breed?
        valid_df["label"].value_counts().plot.bar(figsize=(20, 10)).title(" Valid Dataset");

        # print
        print(f" For Valid Total images: {len(valid_df)} and Total labels: {len(classification)}")
        
      except:
        print(f"No folder {valid_image_path} found")

        # empty dataframe
        valid_df = pd.DataFrame({
          
          'image' : [],
          'label' : []

        })

   # for test path
  if test_path:
    # loop through each folder
    image_data_test = []
    label_data_test = []

    for test_image_path in classification:
      try:
        for image in listdir(test_path+"/"+test_image_path):
          image_data_test.append(test_path+"/"+test_image_path+"/"+image)
          label_data_test.append(test_path)
  
        test_df = pd.DataFrame({
          'image' : image_data_test,
          'label' : label_data_test
           })
        
        # How many images are there of each breed?
        test_df["label"].value_counts().plot.bar(figsize=(20, 10)).title(" Test Dataset");

        print(f" For Test Total images: {len(test_df)} and Total labels: {len(classification)}")  
      except:
        print(f"No folder {test_image_path} found")

        # empty dataframe
        test_df = pd.DataFrame({
          
          'image' : [],
          'label' : []
        })
   
  # before process
  if before_process:
    # if numpy return True then return X and y in numpy array
    if numpy_return:
      return classification, train_df['image'].to_numpy(), train_df['label'].to_numpy(), valid_df['image'].to_numpy(), valid_df['label'].to_numpy(), test_df['image'].to_numpy(), test_df['label'].to_numpy()
    # return data dataframe combine
    elif main_data_return:
      return classification, train_df, valid_df, test_df
    # return image and label in dataframe
    else:
      return classification, train_df['image'], train_df['label'], valid_df['image'], valid_df['label'], test_df['image'], test_df['label']

   # not before process
  else:
    return classification, create_data_batches(train_df['image'], train_df['label'],batch_size=BATCH_SIZE), valid_df['image'], valid_df['label'], test_df['image'], test_df['label']




