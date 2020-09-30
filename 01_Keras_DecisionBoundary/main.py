########################################################################
# Import packages
########################################################################
import numpy as np
np.random.seed(0)  #for reproducibility
from sklearn import datasets, preprocessing
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.colors
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization
from tensorflow.keras.optimizers import SGD


########################################################################
# Load data
########################################################################
def load_data():
    # Load dataset from scikit-learn
    X, Y = datasets.make_moons(n_samples=1000, noise=0.1, random_state=0)    
    
    # Split into train and test sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)

    return X_train, X_test, Y_train, Y_test


########################################################################
# Create model
########################################################################
def create_model(X_train, X_test, Y_train, Y_test):

    # Defines kwarg dict for convenience
    layer_kw = dict(activation='relu')
    layer_kw2 = dict(activation='sigmoid')
    
    # Generate sequential model
    model = Sequential([
        Dense(5, input_shape=(2, ), **layer_kw),
        BatchNormalization(),
        Dense(5, **layer_kw),
        Dense(1, **layer_kw2),
    ])
    
    # Print model summary
    model.summary()

    # Compile model
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-2), loss='binary_crossentropy', metrics=["accuracy"])

    # Fit the model, defines minibatch and number of epochs
    batch_size = 200
    epochs = 200
    verbose = 1
    model.fit(X_train,Y_train, validation_data=(X_test,Y_test), batch_size=batch_size, verbose=verbose, epochs=epochs)
    return model
    

########################################################################
# Plot decision boundary
########################################################################
def plot_decision_boundary(X_train, X_test, Y_train, Y_test, model):

    # Define region of interest by data limits to plot
    xmin, xmax = X_train[:,0].min() - 1, X_train[:,0].max() + 1
    ymin, ymax = X_train[:,1].min() - 1, X_train[:,1].max() + 1
    steps = 100
    x_span = np.linspace(xmin, xmax, steps)
    y_span = np.linspace(ymin, ymax, steps)
    xx, yy = np.meshgrid(x_span, y_span)
    
    # Flatten and concatenate xx and yy
    X_points_to_plot = np.c_[xx.ravel(), yy.ravel()]

    # Make predictions across region of interest
    Y_predicted = model.predict(X_points_to_plot)

    # Plot decision boundary in region of interest
    z = Y_predicted.reshape(xx.shape)


    # Create a binary colormap (red and blue)
    cmap = matplotlib.colors.ListedColormap(['red', 'blue'])
    boundaries = [0, 0.5, 1]
    norm = matplotlib.colors.BoundaryNorm(boundaries, cmap.N, clip=True)


    # Create filled contour plot
    plt.figure(1)
    fig, ax = plt.subplots()
    cax = ax.contourf(xx, yy, z, cmap=cmap, levels=[0,0.5,1], norm=norm, alpha=0.5)
    fig.colorbar(cax,ticks=[0, 1])


    # Get predicted labels on train and test data as well and plot points (scatter plot)
    # Train data (has labels already)
    ax.scatter(X_train[:,0], X_train[:,1], c=Y_train, cmap=cmap, lw=0)
    
    # Test data (has labels already)
    ax.scatter(X_test[:,0], X_test[:,1], c=Y_test, cmap=cmap, lw=0)

    # Show plot (if not in interactive mode)
    plt.savefig('result.pdf')
    plt.show()
    
    
    
########################################################################
# Call functions
########################################################################
if __name__ == "__main__":
    X_train, X_test, Y_train, Y_test = load_data()
    model = create_model(X_train, X_test, Y_train, Y_test)
    plot_decision_boundary(X_train, X_test, Y_train, Y_test, model)
    
