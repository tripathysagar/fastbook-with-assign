1. What problem does collaborative filtering solve?<br/>
   It helps us building recomadation system for any user and prroduct pair. help/find similarity between users and products. 
1. How does it solve it?<br/>
   by buiding embeding vector of all the products and user so that there features are encoded, known as **latent factors**. If two user like a product and then  we can recomend user1's best liked product to user 2.
1. Why might a collaborative filtering predictive model fail to be a very useful recommendation system?<br/>
   When dataset is skewed. If the dataset is dominaated by product of  certain type then the new user will only recommended only that type. No new product will be recommended.Leading to bad feed back loop.
1. What does a crosstab representation of collaborative filtering data look like?<br/>
   A table of user and movie. Where each entry in the table will be the raiting of a movie if given .
1. Write the code to create a crosstab representation of the MovieLens data (you might need to do some web searching!).
1. What is a latent factor? Why is it "latent"?<br>
    in a dataset the products/users have similar in nature. A user's movie taste and movies's nature. It is hidden and we have to learn it.

1. What is a dot product? Calculate a dot product manually using pure Python with lists.<br>
    ```
    arr1 = [1,2,3]
    arr2 = [1,2,3]
    sum([arr1[i] * arr2[i] for i in range((3))])
    ```
1. What does `pandas.DataFrame.merge` do?<br>
    Joins two table on a perticular  culumn name. If a same coulmn is presnt in both the table it makes the join on that. Just like SQL `JOIN`.

1. What is an embedding matrix?<br>
    N dimensional vector representation of all the unique data points is known as embedding matrix. 
1. What is the relationship between an embedding and a matrix of one-hot-encoded vectors?<br>
    We can extract the peticular item's embeding by doing the matrix multiplication.
1. Why do we need `Embedding` if we could use one-hot-encoded vectors for the same thing?<br>
    One hot encoding is information sparse. For a given N dimentional vector there are `N-1` O's in  the vector.
1. What does an embedding contain before we start training (assuming we're not using a pretained model)?<br>
    it contains randomly initialized number
1. Create a class (without peeking, if possible!) and use it.<br>
    ```
    class MyIterator:
        def __init__(self, data):
            self.data = data
            self.index = 0

        def __str__(self):
            return f"*****{self.data}*****"
        def __iter__(self):
            return self

        def __next__(self):
            if self.index >= len(self.data):
                self.index=0
                raise StopIteration
            result = self.data[self.index]
            self.index += 1
            return result

    my_iter = MyIterator([1, 2, 3, 4, 5])
    for item in my_iter:
        print(item)
    ```
1. What does `x[:,0]` return?<br>
    Extract first column of the `x`
1. Rewrite the `DotProduct` class (without peeking, if possible!) and train a model with it.<br>
    ```
    class DotProd(Module):
        def __init__(self, n_users, n_movies, n_internal, y_limit=(0,5.5)):  
  
            self.user_embd = Embedding(*n_users)
            self.movie_embd = Embedding(*n_movies)

            self.net = nn.Sequential(
                nn.Linear(n_users[1] + n_movies[1], n_internal),
                nn.ReLU(),
                nn.Linear(n_internal, 1)
            )

            self.y_range = y_limit
  
        def forward(self, x):
            x = self.user_embd(x[:,0]),  self.movie_embd(x[:,1])
            x = torch.cat(x, dim=1)
            return sigmoid_range(self.net(x), *self.y_range)
    ```


1. What is a good loss function to use for MovieLens? Why? <br>
    CrossEntropy loss can be good as the movie rating is discreate data(0,5). L1 loss can also be used it behaves similar to MSE loss.
1. What would happen if we used cross-entropy loss with MovieLens? How would we need to change the model?<br>
    The model will predict the probability of a rating.<br>
    We have to use one hot encoding for  the ratings, the output needed to pass through a softmax layer for getting all the probbility to 1. Then a argmax layer for classification.
1. What is the use of bias in a dot product model?<br>
    When wight are near zero. There might be a feature of a movie or user needed to be learnt
1. What is another name for weight decay?<br>
    `L2 regularization`
1. Write the equation for weight decay (without peeking!).<br>
    ```
    loss_wd = loss + w * (parameters ** 2).sum()
    ```
1. Write the equation for the gradient of weight decay. Why does it help reduce weights?<br>
    ```
    parameters.grad = 2 * parameters
    ```
1. Why does reducing weights lead to better generalization?<br>
    it limits the over fitting. Lowers the peak.
1. What does `argsort` do in PyTorch?<br>
    Find the index of each element wrt all the element
1. Does sorting the movie biases give the same result as averaging overall movie ratings by movie? Why/why not?<br>
    **TODO**
1. How do you print the names and details of the layers in a model?<br>
    `learn.model`
1. What is the "bootstrapping problem" in collaborative filtering?
1. How could you deal with the bootstrapping problem for new users? For new movies?<br>
1. How can feedback loops impact collaborative filtering systems?<br>

1. When using a neural network in collaborative filtering, why can we have different numbers of factors for movies and users?<br>
    If the no of user is more or less than the no of movies then no of factors for encoding might increase or decrease repectively. As the no of user in  the dataset is high we are using more to encode than movies.
1. Why is there an `nn.Sequential` in the `CollabNN` model?<br>
    to use deep-NN on top of embedding
1. What kind of model should we use if we want to add metadata about users and items, or information such as date and time, to a collaborative filtering model?<br>
    Random Forest