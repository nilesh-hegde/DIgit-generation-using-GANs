import tensorflow as tf
import matplotlib.pyplot as plt

'''
Arguments - GAN model, MNSIT dataset, batch size, number of epochs
Return - None
Description - This function starts execution and implements the extension by calling train_gan function
'''
def train_gan(gan, dataset, batch_size, codings_size, n_epochs = 10):
    # Get layers for individual models of GAN
    generator, discriminator = gan.layers
    
    #Train the model
    for epoch in range(n_epochs):
        print("Epoch : {}".format(epoch))
        for X_batch in dataset:
            # Generate new images
            noise = tf.random.normal(shape = [batch_size, codings_size])
            generated_images = generator(noise)
            X_batch = tf.cast(X_batch, tf.float32)
            X_fake_and_real = tf.concat([generated_images, X_batch], axis = 0)
            y1 = tf.constant([[0.]] * batch_size + [[1.]] * batch_size)
            
            # Discimnator trying to distinguish
            discriminator.trainable = True
            discriminator.train_on_batch(X_fake_and_real, y1)
            noise = tf.random.normal(shape = [batch_size, codings_size])
            y2 = tf.constant(([[1.]] * batch_size))
            
            # Discrimator not trainable now
            discriminator.trainable = False
            gan.train_on_batch(noise, y2)
        
        # Plot the generated images
        plt.figure(figsize=(6,6))
        for i in range(32):
            plt.subplot(8, 4, 1+i)
            plt.axis('off')
            plt.imshow(generated_images[i], cmap = 'gray')
        plt.savefig('myGAN/{}_epoch.jpg'.format(epoch))
    return

'''
Arguments - None
Return - None
Description - This function starts execution and implements the extension by calling train_gan function
'''
def main():
    # Read MNSIT data and preprocess
    mnist = tf.keras.datasets.mnist
    (X_train, Y_train),(X_test, Y_test) = mnist.load_data()
    X_train = X_train / 255
    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices(X_train).shuffle(1000)
    dataset = dataset.batch(batch_size, drop_remainder = True).prefetch(1) 
    codings_size = 30
    
    # Create generator
    generator = tf.keras.models.Sequential([
                    tf.keras.layers.Dense(128,activation = 'selu', input_shape = [codings_size]),
                    tf.keras.layers.Dense(256, activation = 'selu'),
                    tf.keras.layers.Dense(28*28, activation = 'sigmoid'),
                    tf.keras.layers.Reshape([28,28])
    ])
    
    # Create discriminator
    discriminator = tf.keras.models.Sequential([
                    tf.keras.layers.Flatten(input_shape = [28,28]),
                    tf.keras.layers.Dense(256, activation = 'selu'),
                    tf.keras.layers.Dense(128, activation = 'selu'),
                    tf.keras.layers.Dense(1, activation = 'sigmoid')
    ])
    
    # Combine discrimator and generator to form GAN model and train
    gan = tf.keras.models.Sequential([generator, discriminator]) 
    discriminator.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop')
    
    # Initially discriminator is not trainable
    discriminator.trainable = False
    gan.compile(loss = 'binary_crossentropy', optimizer = 'rmsprop') 
    train_gan(gan, dataset, batch_size, codings_size)
    return

if __name__ == "__main__":
   main()
