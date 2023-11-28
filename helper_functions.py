
# Note: The following confusion matrix code is a remix of Scikit-Learn's
# plot_confusion_matrix function - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.plot_confusion_matrix.html
import itertools
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
# import tensorflow as tf
import pandas as pd

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



def binary_plot_loss_history_confiusion_matrix(model,pl_history, pl_X_test, pl_y_test, 
                                               classes=None,lost_fig_size=(10, 10), figsize=(10, 10), 
                                               text_size=15, norm=True, savefig=False,
                                               save_model=False,save_path=""):
  
  """ Make skelearn report with loass and accuracy plot with learning rate if preset
  and confusion matrix and loss history also showing higest validation accuracy against learning rate also
  option to save model and confusion matrix to file
  
  Arug:
    model: model
    pl_history: history of model
    pl_X_test: test data input
    pl_y_test: test data output
    classes: classes if avilable else None
    lost_fig_size: size of lossand accuracy plot
    figsize: size of confusion matrix
    text_size: size of text in confusion matrix
    norm: normalize confusion matrix with % result in confusion matrix
    savefig: save confusion matrix to file (default=False)
    save_model: save model( default=False)
    save_path: path to save model (default="")
    """


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

  
  

