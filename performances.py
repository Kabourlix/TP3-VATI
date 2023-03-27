import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import seaborn as sns
from tensorflow import keras
from sklearn.metrics import make_scorer, confusion_matrix, cohen_kappa_score, classification_report
from sklearn.model_selection import cross_val_score

#history = model.fit(X_train,y_train,verbose=1,epochs=50, batch_size=64)

def show_performance(model,history,X_test,y_test):
    # Evaluation du modele
    score = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test loss: {score[0]:4.4f}")
    print(f"Test accuracy: {score[1]:4.4f}")

    # prediction
    y_pred = np.argmax(model.predict(X_test), axis=-1)

    # Trace les courbes d'apprentissage et de validation
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'y', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # accuracy graph
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    plt.plot(epochs, acc, 'y', label='Training accuracy')
    plt.plot(epochs, val_acc, 'r', label='Validation accuracy')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

    # Compute the Cohen's kappa coefficient
    y_true = y_test[:, -1]
    kappa = cohen_kappa_score(y_true, y_pred)
    # Print the kappa score
    print("Cohen's kappa coefficient: ", kappa)

    # affiche le rapport de classification
    print("Results of the test set:")
    print(classification_report(y_test, y_pred))

    # affiche la matrice de confusion
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True)
    plt.show()
