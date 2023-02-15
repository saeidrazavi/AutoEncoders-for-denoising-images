# AutoEncoders for denoising images

## what are AutoEncoders?

AutoEncoder is an unsupervised learning Algorithm used typically for dimensionality reduction (data compression), AutoEncoder is a Neural Network with a specific Architecture like bellow :

![1](https://user-images.githubusercontent.com/67091916/219174340-547b4992-4f7b-45cd-8b04-9ba82c5a9778.PNG)

in this repo, we use images of mnist dataset, add guassian noise to them at first, and then make a nerual network using pythorh, to denoise the noisy images 

### how to add noise to images? 


```python
class GaussianNoise(object): 
    def __init__(self, mean=0., std=0.1): 

        self.std = std
        self.mean = mean

    def __call__(self, tensor): 

        return tensor + torch.randn(tensor.size()) * self.std + self.mean
```
 after making above class, by using `transform.compose` we can convert images into tensor first, and then add guassian noise to them
 
 ```python
noisy_data = datasets.MNIST(
    root = 'data',
    train = True,                         
    transform=transforms.Compose([
                       transforms.ToTensor(),
                       GaussianNoise()]), 
    download = True,            
)
```

### build an AutoEncoder using pythorch 
The key idea behind AutoEncoder is to use to basic transforms. `nn.linear` and `nn.Relu` untill we reach  the latent vector. after that we again use these two transforms to reach reconstructed form of original image.

- Implementation
```python
   class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder , self).__init__()        
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 16) ,
            nn.ReLU(),
            nn.Linear(16, 10)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(10, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded_data = self.encoder(x)
        decoded_data = self.decoder(encoded_data)
        return decoded_data

 

```

if we plot one image for each digit and visualize the initial images, their corresponding noisy images and decoded images we have: 

![2](https://user-images.githubusercontent.com/67091916/219180310-a22de70c-e405-45cf-8ff5-de7e8769ab58.png)

### latent vectors for 10 random digits  

![3](https://user-images.githubusercontent.com/67091916/219181957-e39ac79a-6c86-4abd-adb2-64bd5cec0ae0.png)

### using dimensionality reduction methods(PCA and TSNE) to visualize latent vectors in 2D space

we expect that, if we train good neural network model, each digit, cover individual area and we reach segmented 2D space without interference.

#### PCA method : 

![4](https://user-images.githubusercontent.com/67091916/219193089-6083eb8d-2ba8-4e0f-94b4-f0ca81d6b698.png)

#### TSNE method : 

![5](https://user-images.githubusercontent.com/67091916/219193054-dc0cb7a6-3fb1-47cc-8a8d-350896204630.png)

 ### Conclustion : 
 In this problem, TSNE is more powerful in transforming our latent vectors into 2d space.  as we can see above, we have segmented space in TSNE but for PCA, it's not very clear and separated
