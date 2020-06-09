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
# Domain Background:
## Recurrent Neural Networks:
![](/images/AI Existentialism/RNNUnfolded.png){:height="80%" width="80%"}
<div style="text-align:center">The image is a Simple Recurrent Network or Elman Network</div>

### Defintions:
_x<sub>t</sub> : input vector at time-step t<br>
h<sub>t</sub> : hidden layer vector (hidden state) at time-step t<br>
o<sub>t</sub> : output vector at time-step t_<br>

_b<sub>h</sub> : bias used in the creation of the next hidden state<br>
b<sub>o</sub> : bias used for the output_<br>

_U : weight matrix from input to the hidden state<br>
V : weight matrix from hidden state to the next hidden state<br>
W : weight matrix from hidden state to output_<br>

All weight matrices and biases stay the same throughout each time step.

### Mathematical View:
_x<sub>t</sub> ∈ R<sup>n</sup><br>
h<sub>t</sub> ∈ R<sup>d</sup><br>
b<sub>h</sub> ∈ R<sup>d</sup><br>
b<sub>o</sub> ∈ R<sup>k</sup>_<br>

_U ∈ R<sup>d x n</sup><br>
V ∈ R<sup>d x d</sup><br>
W ∈ R<sup>k x d</sup><br>
(W can be the weight matrix that connects h<sub>t</sub> to a fully connected layer)_

_Φ : R→R<br>
(Activation Function between layers: tanh, relu, or sigmoid)_
