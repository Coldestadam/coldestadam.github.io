---
title: 'AI Existentialism'
subtitle: 'Using LSTMs that was trained on works of Kierkegaard, Nietzsche, Marx for the purpose of text-generation'
date: 2020-5-28
description: 'Using LSTMs that was trained on works of Kierkegaard, Nietzsche, Marx for the purpose of text-generation'
featured_image: '/images/AI Existentialism/friedrich-nietzsche-19.jpg'
---

![](/images/AI Existentialism/nietzsche_main.jpg)

<div style="text-align:center">
  <a href="https://github.com/Coldestadam/Existentialism-Text-Generation" class="button button--large">View Code Here</a>
</div>
<br>

# Project Goal:
In this course, we have learned how to create Statistical/ML models that use non-sequential data, such as images or variables for the use of regression tasks. These are considered non-sequential since the dataset presumes that each sample share no relationship in terms of time. Sequential data presumes that the samples are connected through time, and text is an example of this. An example is this sentence, “The doctor recommends to eat an apple a day.”, here we see that the verb ‘eat’ connects the subject ‘doctor’ and the object ‘apple’.

In this project, we explore sequential modeling more in-depth and how we can predict the next word from a sequence of words. For example, if the sequence is, “In this morning, I usually drink \_\_\_\_”, the model should predict ‘coffee’ or ‘tea’. I will gather great works from different authors and have the model learn from the text for the use of text generation.

---
# Domain Background
## Recurrent Neural Networks
![](/images/AI Existentialism/RNNUnfolded.png){:height="80%" width="80%"}
<div style="text-align:center">The image is a Simple Recurrent Network or Elman Network</div>

### Defintions:
_**x<sub>t</sub> : input vector at time-step t<br>
h<sub>t</sub> : hidden layer vector (hidden state) at time-step t<br>
o<sub>t</sub> : output vector at time-step t**_<br>

_**b<sub>h</sub> : bias used in the creation of the next hidden state<br>
b<sub>o</sub> : bias used for the output**_<br>

_**U : weight matrix from input to the hidden state<br>
V : weight matrix from hidden state to the next hidden state<br>
W : weight matrix from hidden state to output**_<br>

All weight matrices and biases stay the same throughout each time step.

### Mathematical View:
_**x<sub>t</sub> ∈ R<sup>n</sup><br>
h<sub>t</sub> ∈ R<sup>d</sup><br>
b<sub>h</sub> ∈ R<sup>d</sup><br>
b<sub>o</sub> ∈ R<sup>k</sup>**_<br>

_**U ∈ R<sup>d x n</sup><br>
V ∈ R<sup>d x d</sup><br>
W ∈ R<sup>k x d</sup>**<br>
(W can be the weight matrix that connects h<sub>t</sub> to a fully connected layer)_

_**Φ : R→R**<br>
(Activation Function between layers: tanh, relu, or sigmoid)_

1. Getting the hidden state at time-step _t_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_**h<sub>t</sub> = Φ(b+Vh<sub>t-1</sub> +Ux<sub>t</sub>)**_
2. Applying a fully connected layer at each time step<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_**o<sub>t</sub> = Φ(b<sub>o</sub> + Wh<sub>t</sub>)**_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;(Usually, the activation function here will be softmax if you
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; are predicting k-classes at each time step)
3. After calculating the output, the network will repeat these steps for the next time-step

### Deep RNNs
![](/images/AI Existentialism/deep_rnn.png){:height="50%" width="50%"}
<div style="text-align:center">The image is a Multiple-Layer RNN</div>

Just as deep neural networks have multiple layers, RNNs can also have multiple layers. If the number of layers increases, the model can learn more interesting patterns. However, it becomes more computationally expensive, so you must balance your resources with the amount of time you would like to train the model. The intuition is the same, it just increases the number of hidden states per time-step.

## Vanishing and Exploding Gradients
The backpropagation algorithm used in RNNs is different from feed-forward neural networks since each gradient of the loss has a dependency on all the inputs in the sequence through time. The method of backpropagation is called Backpropagation Through Time or BPTT for short. To explain this more in-depth, [here](https://www.youtube.com/watch?v=SEnXr6v2ifU&feature=youtu.be&t=1230) is a lecture that was given at MIT that explains it quite well. I encourage you to watch the entire lecture if you have time, it is great at giving the fundamentals of RNNs.

Vanishing Gradients are when the gradients get close to 0 as they are calculated through time. Simple RNNs, such as the one we have looked at, have vanishing gradients over a certain number of time-steps in the sequence. Therefore Simple RNNs are not able to have Long-Term Dependencies after a certain number of time-steps, or rather it has difficulty remembering the data from the beginning of the sequence. Gated RNNs such as LSTMs and GRUs have risen to tackle the Vanishing Gradient problem, and LSTMs are used in this project.

![](/images/AI Existentialism/gradient_clipping.png){:height="60%" width="60%"}
<div style="text-align:center">Figure 10.17 from Section 10.11.1 of Deep Learning Book</div>

Exploding Gradients are when the gradients become too large, that when an optimization step occurs, it can step so far that the model will start to converge in a different region. The solution for Exploding Gradients is to clip the gradients to prevent it from exploding. One option is to clip the norm of the gradient, and you can read more about this in [Deep Learning Book section 10.11.1](https://www.deeplearningbook.org/contents/rnn.html) by Ian Goodfellow.

## Solution of LSTMs
![](/images/AI Existentialism/LSTM.png){:height="80%" width="80%"}
<div style="text-align:center">A Look into LSTM cell and it’s gates</div>

As said before, gated RNNs are a solution to the Vanishing Gradient problem. LSTMs and GRUs are gated RNNs, and they attempt to retain information throughout a sequence. In each LSTM cell, it outputs both a cell state and a hidden state, the hidden state reacts as normally but the cell state only transfers through the cells. What LSTMs do is to learn what information needs to be passed through into the rest of the sequence and what information needs to be restricted.

There are four gates in the LSTM:

1. Forget Gate - Takes the input and previous hidden state and determines what to remember<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_**f<sub>t</sub> = σ(W<sub>f</sub> · \[h<sub>t-1</sub> , x<sub>t</sub> ] + b<sub>f</sub>)**_
2. Learn Gate - Decides what new information will be stored in the cell state<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_**i<sub>t</sub> = σ(W<sub>i</sub> · \[h<sub>t-1</sub> , x<sub>t</sub> ] + b<sub>i</sub>)**_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_**N<sub>t</sub> = tanh(W<sub>c</sub> · \[h<sub>t-1</sub> , x<sub>t</sub> ] + b<sub>c</sub>)**_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_**Output: N<sub>t</sub>i<sub>t</sub>**_
3. Remember Gate - Brings Learn Gate and Forget Gate to update the cell state<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_**C<sub>t</sub> = C<sub>t-1</sub> * f<sub>t</sub> + N<sub>t</sub> * i<sub>t</sub>**_
4. Use Gate - Outputs the hidden state<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_**o<sub>t</sub> = σ(W<sub>o</sub> ·\[h<sub>t-1</sub> , x<sub>t</sub> ] + b<sub>o</sub>)**_<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;_**h<sub>t</sub> = o<sub>t</sub> * tanh(C<sub>t</sub>)**_

## Skip-Gram Word2Vec
![](/images/AI Existentialism/skipgram.png){:height="100%" width="100%"}
<div style="text-align:center">Skip-gram Word2Vec</div>

One of the issues with early NLP problems were the dimensionality of representing individual words as a one-hot encoded vector. The problem is that they can hold a lot of unnecessary memory and can be computationally heavy when weights are connected to the vector. What Skip-gram Word2Vec does is that it outputs a unique vector for each word and decreases the dimensionality. It does this through an embedding layer or the weights of a feed-forward network. We can get a unique representation of a word through the rows in the weight matrix in a feed-forward network because the input is a one-hot encoded vector being multiplied against a weight matrix. This will just output the row of the weight matrix that corresponds with the 1 in the input. There is a great conceptual overview [here](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/) and the original paper is [here](https://arxiv.org/abs/1301.3781).

---

# Creating my LSTM
![](/images/AI Existentialism/many_to_one_rnn.png){:height="100%" width="100%"}
<div style="text-align:center">Many to One RNN</div>

From the image above, my neural network will be structured like this. Each input for a time-step will be a word, and the output _y_ will be the next word after time-step ​t​ in the sequence. Using the same example from the beginning of the paper, our sequence will be:

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sequence = \[In, this, morning, I, usually, drink]
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Y = \[coffee]

Using Cross-Entropy Loss, we will measure the difference between the predicted output with the correct output, in which the network will learn the patterns between the input and the output. There are outputs given at every time-step, however, we will only focus on the last time-step _t_, where we take the loss and backpropagate from there back into the network.

---

# Data

## First Dataset

The first dataset includes these Great Works:
1. Karl Marx's [Communist Manifesto](https://www.gutenberg.org/ebooks/61)
2. Friedrich Wilhelm Nietzsche's [Beyond Good and Evil](https://www.gutenberg.org/ebooks/4363)
3. Friedrich Wilhelm Nietzsche's [Thus Spoke Zarathustra](https://www.gutenberg.org/ebooks/1998)
4. Søren Kierkegaard's [Selected Writings, including Fear and Trembling](https://www.gutenberg.org/ebooks/60333)

After getting the texts online, I hand-cleaned it to remove repetitive new lines, titles of chapters, commentaries, footnotes, and anything that did not contain the writing of the authors. After cleaning the texts, I concatenated the texts together and then began preprocessing it for the model. Theoretically, since all of the data is used to train the network, the proportion of how the model will be influenced by the authors is equal to the proportion of the author’s writings in the dataset. Also, the disadvantage of this dataset is that the style of writing between works is different so the generated text will not be as coherent.

