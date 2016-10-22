---
layout: post
title: There's nothing quite like Neural Networks 
published: true
---

[Slides](https://nbviewer.jupyter.org/github/mattilyra/notebooks/blob/master/pydata-hackathon-report.ipynb)

The slides for my short presentation on failing to use neural networks presented at PyData Berlin (19.10.2016). The aim was to continue the Felix Biessmann's work at a PyData Berlin hackevent by applying neural networks to the same datasets the Felix had already analysed using other machine learning methods. For some background on Felix's work on detecting political bias in text see his talk at the PyData Berlin 2016 conference [youtube](https://www.youtube.com/watch?v=IhUSiXXg4rg).

- [keras.io](http://keras.io)
- [keras github](https://github.com/fchollet/keras)
- Character-Level Convolutional Networks for Text Classification (Zhang et. al, NIPS 2015)
  - [arxiv](http://arxiv.org/abs/1509.01626)
  - [github](https://github.com/zhangxiangxiao/Crepe)
  - [Notes by Adriaan Schakel (PDF)](asserts/nextstep.pdf) - published under his permission
  - [Discussion on how to implement the model in keras](https://github.com/fchollet/keras/issues/233)
- Convolutional Neural Networks for Text Classifiation (Kim, EMNLP 2014)
  - [arxiv](https://arxiv.org/abs/1408.5882)
  - [github](https://github.com/yoonkim/CNN_sentence)
- [WildML](http://wildml.com)
- [deeplearning.net CNN](http://deeplearning.net/tutorial/lenet.html)



----

_Footnote_ 

_I'd like to thank Adriaan Schakel and David Batista for their comments and additional resources before and after the talk. As pointed out by Adriaan during the questions, my aim was not and is not to trash talk `keras` but to point out that it was rather difficult to switch from more traditional ML methods (SVM, NB) to neural networks due to the complexities involved in understanding the composition of said networks. It is often said that with neural networks you don't need domain experts anymore because the network will do the work of the experts for you, I wanted to point out that you do still need domain experts just in a slightly different domain._
{: font-color=gray }
