---
layout: post
title: Evaluating Topic Models
published: true
description:
---

<h1>Evaluating Topic Models</h1>
<p><em>I gave a talk on evaluating topic models at the 2017 PyData Berlin conference, this post is an expansion to that talk <a href="https://www.youtube.com/watch?v=UkmIljRIG_M">https://www.youtube.com/watch?v=UkmIljRIG_M</a> - there is also a <a href="https://nbviewer.jupyter.org/github/mattilyra/pydataberlin-2017/blob/master/notebook/EvaluatingUnsupervisedModels.ipynb">notebook</a> with all the model training details should you wish to run the models yourself . </em></p>
<p>Often evaluating topic model output requires an existing understanding of what should come out. The output should reflect our understanding of the relatedness of topical categories, for instance <strong>sports</strong>, <strong>travel</strong> or <strong>machine learning</strong>. Topic models are often evaluated with respect to the semantic coherence of the topics based on a set of top words from the topics and some reference corpus. It is not clear if a set of words such as <code>{cat, dog, horse, pet}</code> captures the semantics of animalness or petsiness fully.</p>
<p>This post explores a number of these issues in context and aims to provide an overview of the research that has been done in the past 10 or so years, mostly focusing on topic models.</p>
<p>There are three parts to this post</p>
<ol>
<li>Eye Balling models
<ul>
<li>ways of making visual, manual inspection of models easier</li>
</ul>
</li>
<li>Intrinsic Evaluation Methods
<ul>
<li>how to measure the internal coherence of topic models</li>
</ul>
</li>
<li>Putting a Number on Human Judgements
<ul>
<li>quantitative methods for evaluating human judgement</li>
</ul>
</li>
</ol>
<p>I am also ignoring one obivous evaluation method: evalution at an extrinsic task. If the topic model is used to perform classification then the classification metric is quality measure for the topic model.</p>
<hr />
<h1>Why Evaluate Models</h1>
<p>We would like to be able to say if a model is objectively good or bad, and compare different models to each other. This requires us to have an objective measure for the quality of the model but many of the tasks mentioned above require subjective evaluation.</p>
<p>In practical applications one needs to evaluate if "the correct thing" has been learned, often this means applying implicit knowledge and "eye-balling". Documents that talk about <em>football</em> should be in the same category for instance. Ideally this information should be captured in a single metric that can be maximised. It is not clear how to formulate such a metric however, over the years there has been numerous attempts from various different angles at formulating semantic coherence, none capture the desired outcome fully and there are issues one should be aware of in applying those metrics.</p>
<p>Some of the issues are related to the metrics being used or issues one should be aware of when applying those metrics, but others are related to external factors, like which kind of held out data to use. Natural language is messy, ambiguous and full of interpretation, that's where a lot of the expressive richness comes from. Sometimes trying to cleanse the ambiguity also reduces language to an unnatural form.</p>
<hr />
<h1>Fake News</h1>
<p>In order to evaluate a model, we must of course have one. I'll use the same model(s), built from the Fake News data set on Kaggle, throughout this notebook. If you want to train your own model you can download the dataset here <a href="https://www.kaggle.com/mrisdal/fake-news">https://www.kaggle.com/mrisdal/fake-news</a>, you can refer to <a href="https://github.com/mattilyra/pydataberlin-2017/blob/master/notebook/EvaluatingUnsupervisedModels.ipynb">this notebook</a> for code samples on how to train the model.</p>
<p>There is a total of 12357 non empty english language documents, should be enough to build a model. I preprocessed the data using <code>spacy</code>, getting rid of some non content words and chucked that into <code>gensim</code> to train a 35 topic model (and later a 100 topic model). Inspecting the top 6 six words from each topic in the model we can certainly identify some structure, below is a small sample. There are topics about the Flint Michigan water crisis (Topic 11), the Dakota Access Pipeline (Topic 9) protests and the US elections.</p>
<table class="dataframe" border="1">
<thead>
<tr>
<th></th>
<th></th>
<th colspan="6">Top Words</th>
</tr>
<tr>
<th></th>
<th></th>
<th>0</th>
<th>1</th>
<th>2</th>
<th>3</th>
<th>4</th>
<th>5</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="6" valign="top">Topics</th>
<th>7</th>
<td>use</td>
<td>drug</td>
<td>health</td>
<td>company</td>
<td>report</td>
<td>medical</td>
</tr>
<tr>
<th>8</th>
<td>post</td>
<td>comment</td>
<td>facebook</td>
<td>news</td>
<td>video</td>
<td>result</td>
</tr>
<tr>
<th>9</th>
<td>pipeline</td>
<td>dakota</td>
<td>police</td>
<td>water</td>
<td>rock</td>
<td>protester</td>
</tr>
<tr>
<th>10</th>
<td>child</td>
<td>school</td>
<td>police</td>
<td>family</td>
<td>when</td>
<td>people</td>
</tr>
<tr>
<th>11</th>
<td>water</td>
<td>study</td>
<td>health</td>
<td>toxic</td>
<td>flint</td>
<td>tea</td>
</tr>
<tr>
<th>12</th>
<td>clinton</td>
<td>campaign</td>
<td>hillary</td>
<td>email</td>
<td>podesta</td>
<td>wikileaks</td>
</tr>
</tbody>
</table>
<h1>Eye Balling</h1>
<p>As the unsupervised models still come with hyperparameters you need to have some way of evaluating if and when you've set them <em>correctly</em>, furthermore, how do you know if model A is better than model B. It is often easiest to start by just looking at the model output to find out if what has been learned corresponds to your prior expectation of what should be learned. Evaluating model quality by inspecting the top words from each topic is labour intensive and quite difficult for larger models. Luckily there are a number of research projects that have looked at better ways of showing the semantic regularities of a model.</p>
<h3>Termite</h3>
<p>The visualisations in the <code>Termite</code> <a href="http://vis.stanford.edu/papers/termite">paper</a> look very promising, but the code is a little cumbersome to run. The project hasn't been updated in a while and it is all in python 2. The original project has also been split into two separate projects a <em>data server</em> and a <em>visualisation client</em>.</p>
<p>Unfortunately the data server uses an unknown data format in SQLite databases, and the host server where the data sets ought to be is not operational anymore and the project hasn't been maintained since 2014. The project also relies on <code>web2py</code> which at the moment only supports python 2 and there doesn't seem to be any interest in porting it to python 3.</p>
<p>Either way, it is possible to run the project under a python 2 environment with just a few modification to the current development head on github. The changed are needed mostly to conform to changes in the <code>gensim</code> API since 2014. I've forked the project and pushed required changed to <a href="https://github.com/mattilyra/termite-data-server">https://github.com/mattilyra/termite-data-server</a>. Please refer to <a href="https://github.com/mattilyra/pydataberlin-2017/blob/master/notebook/EvaluatingUnsupervisedModels.ipynb">the notebook</a> for instructions on how to run the project and look at the visualisations.</p>
<h3>pyLDAVis</h3>
<p>Some of the work from <code>Termite</code> has been integrated into <code>pyLDAVis</code> which is being maintained and has good interoperability with <code>gensim</code>. Below is an interactive visualisation of the fake news model trained earlier.</p>
<p>For a good description of what you see in the visualisation you can look at the presenation from Ben Mabey, the person who ported the project to Python <a href="https://www.youtube.com/watch?v=tGxW2BzC_DU&amp;index=4&amp;list=PLykRMO7ZuHwP5cWnbEmP_mUIVgzd5DZgH">https://www.youtube.com/watch?v=tGxW2BzC_DU&amp;index=4&amp;list=PLykRMO7ZuHwP5cWnbEmP_mUIVgzd5DZgH</a>.</p>
<p>The data in the visualisation is stored in a JSON file which is about 3.5MB - I've therefore hidden the visualisation by default you can load it by clicking the link <a>show LDAVis</a>.</p>
<p>Comparing the two visualisation one can make some comforting observations. In the bottom left right corner in both visualisation there is a cluster of topics relating to the 2016 U.S. presidential election. The 100 topic model has split the documents up in slightly more specific terms but otherwise both models have captured those semantics and more importantly both visualisations display those topics consistently in a cluster.</p>
<p>Similarly in the visualisation of the 100 topic model the cluster in the top right hand corner is semantically coherent and similar to the cluster in the bottom left hand corner in the visualisation for the 35 topic model. Again both models have captured the Syrian civil war and related issues and consistently placed those topics close together in the topic panel.</p>
<p>The main problem I find the <code>LDAvis</code> is that the spatial dimensions on the left hand side panel are somewhat meaningless.</p>
<p>The area of the circle shows the prevalence of a topic, but visually determining the relative sizes of circles is difficult to do, so while you do get an understanding of which topics are the most important you can't really determine how much more important those topics are compared to the others.</p>
<p>The second problem is the distance between the topics. While the positioning of the topics to some exent preserves semantic similarity allowing some related topics to form clusters, it is a little difficult to determine exactly how similar the topics are. To be fair this is not something that can be blamed on <code>LDAvis</code> as measuring the semantic similarity of topics and then collapsing the multidimensional similarity vectors into 2 dimensions is not an easy task to do. Nevertheless, one shouldn't read too much into the topic distances. Different algorithms for computing the locations - essentially doing multidimensional scaling - will produce wildly different visualisations.</p>
<p>Finally the axes <code>PC1</code> and <code>PC2</code> somehow imply a centrality to the topic space displayed on the left. This naturally begs the question, what do those two axes mean? The problem is that they do now actually mean anything. Technically they are principal components of the collapsed topic space, but that doesn't actually mean anything. Take the 35 topic model with the U.S. politics cluster in the bottom right and the Syrian War cluster in the bottom left corner. Based on the location of the <em>politics</em> cluster one could assume that the <strong><em>meaning</em></strong> of the horizontal axis is politics or perhaps U.S. politics - that however would then mean that the Syrian War is somehow semantically the exact opposite of U.S. politics. That's non-sensical. Furthermore the presence of the axes somehow imply that anything in the middle is void of semantic content, but that isn't true either.</p>
<p>To be clear, I don't find the visualisation as such objectionable, just the presence of the axes. They are misleading and should not be there.</p>
<hr />
<h1>Intrinsic Evaluation</h1>
<p>Perplexity is often used as an example of an intrinsic evaluation measure. It comes from the language modelling community and aims to capture how suprised a model is of new data it has not seen before. This is commonly measured as the normalised log-likelihood of a held out test set</p>
<p>$$ \begin{align} \mathcal{L}(D') &amp;= \frac{\sum_D \log_2 p(w_d;\Theta)}{\mbox{count of tokens}}\\\\ perplexity(D') &amp;= 2^{-\mathcal{L}(D')} \end{align} $$</p>
<p>Focussing on the log-likelihood part, this metric is measuring how probable some new unseen data is given the model that was learned earlier. That is to say, how well does the model represent or reproduce the statistics of the held out data.</p>
<p>Thinking back to what we would like the topic model to do, this makes no sense at all. Let's put aside any specific algorithm for inferring a topic model and focus on what it is that we'd like the model to capture. More often than not the desire is for the model to capture <em>concepts</em> that exist in a particular dataset, concepts that correspond to our notion of how the world is divided into semantic groupings. What is a concept and how can it be represented given the pieces we have?</p>
<p>Let me offer a way of thinking about this that would not pass the mustard in a bachelor's class in philosophy. Luckily we're not in philosophy class at the moment.</p>
<p>Take the following two documents that talk about ice hockey. I've highlighted terms that <strong><em>I</em></strong> think are related to the subject matter, you may disagree with my judgement. Notice that among the terms that I've highlighted as being part of the <em>topic</em> of Ice Hockey are words such as <code>Penguins</code>, <code>opposing</code> and <code>shots</code>. None of these words, on the face of it, would appear to "belong" to Ice Hockey, but seeing them in context makes it clear that <code>Penguins</code> refers to the ice hockey team, <code>shots</code> refers to disk shaped pieces of vulcanised rubber being launched at the goal at various different speeds and <code>opposing</code> refers to the opposing team although it might more commonly be thought to belong to politics or the debate club.</p>
<blockquote>
<p>... began his professional <strong>career</strong> in 1989&ndash;90 with <strong>Jokerit</strong> of the <strong>SM-liiga</strong> and <strong>played</strong> 21 <strong>seasons</strong> in the <strong>National Hockey League</strong> (<strong>NHL</strong>) for the <strong>Winnipeg Jets</strong> ...</p>
<p><strong>Rinne</strong> <strong>stopped</strong> 27 of 28 <strong>shots</strong> from the <strong>Penguins</strong> in <strong>Game</strong> 6 at home Sunday, but that lone <strong>goal</strong> allowed was enough for the <strong>opposition</strong> to break out the <strong>Stanley Cup</strong> <strong>trophy</strong> for the second straight <strong>season</strong>.</p>
</blockquote>
<p>Given the terms that I've determined to be a partial description of Ice Hockey (the concept), one could conceivably measure the coherence of that description by counting how many times those terms occur with each other - co-occur that is - in some sufficiently large reference corpus. Notice however that we're not measuring the coherence of the concept, but of the definition of the concept I just presented.</p>
<p>One of course encounters a problem should the reference corpus never refer to ice hockey. A poorly selected reference corpus could for instance be patent applications from the 1800s, it would be unlikely to find those word pairs in that text, but let's assume that we have a good sample of what modern text looks like and use that to measure the co-occurrences by first looking up co-occurrences of word pairs from the description and then computing some metrics that represents how likely that word pair is given the reference corpus.</p>
<p>This is precisely what several research papers have aimed to do. Take the top words from the topics in a topic model and measure the <em>support</em> for those words forming a coherent concept / topic by looking at the co-occurrences of those term pairs in a reference corpus. Examples of confirmation measures are the <code>UCI</code> and <code>UMass</code> measures, which measure coherence by looking at the pointwise mutual information between two terms and the conditional log likelihood respectively. Specifically</p>
<p>$$ C_{UCI} = \frac{2}{N(N-1)}\sum_{i=1}^{N-1}\sum_{j=i+1}^{N}PMI(w_i, w_j) $$$$ C_{UMass} = \frac{2}{N(N-1)}\sum_{i=2}^{N}\sum_{j=1}^{i-1} \log \frac{P(w_i, w_j) + \epsilon}{P(w_j)} $$</p>
<p>The research was wrapped up into a single paper where the authors develop a <em>coherence pipeline</em>, which allows plugging in all the different methods into a single framework. This <em>coherence pipeline</em> is partially implemented in <code>gensim</code> and is quite straightforward to use. Both the UMass and UCI coherence measures are implemented as well as the vector similarity measure developer by R&ouml;der et. al (see <a href="https://github.com/mattilyra/pydataberlin-2017/blob/master/notebook/EvaluatingUnsupervisedModels.ipynb">notebook</a> for code examples)</p>
<p>The coherence measures are certainly a step in the right direction but they don't completely solve the problem. For instance it's possible that a larger topic model (100 topis) has captured all of the information a smaller (35 topics) model does. The larger model may have captured some additional information and junk, depending on the proportions of junk to additional information the coherence measure can rank the larger model as less coherent - which it in a sense is - even if the non-junk topics it has captured are <em>coherent</em> and informative. It would be trivial for a human to determine which to junk topics are, and consequently ignore those and only use the information from the informative topics. It is therefore very difficult to get away from needing to use both metric evaluations and manual/visual inspection of the models.</p>
<p>I had a long discussion with Lev Konstantinovskiy, the community maintainer for <code>gensim</code> for the past 2 or so years, about the coherence pipeline in <code>gensim</code>. He pointed out that for training topic models coherence is extremely useful as it tends to give a much better indication of when model training should be stopped than perplexity does. Parul Sethi, a Google Summer of Code student for <code>gensim</code>, was kind enough to provide some extra details on the matter. The below notes are from her.</p>
<h2>What Coherence is Useful for (by Parul Sethi)</h2>
<p>While the human judgement for evaluating and comparing topic models by means of visualizations like <code>pyLDAvis</code> could be more appropriate than depending on objective measures like <code>coherence</code> or <code>perplexity</code>, these measures could still be useful in monitoring the training progress of a topic model. For example the graphs below show the value of coherence, perplexity and convergence plotted for every epoch as the LDA training progresses.</p>
<p><img src="gensim_lda_training.png" /></p>
<p>These graphs are produced using gensim and Visdom with LDA model trained on Kaggle&rsquo;s fake news dataset. You can refer to this notebook for instructions to visualize these metrics.</p>
<p>As we can see that the value stops changing after some epochs for all the metrics which could help us decide if our model is sufficiently trained and it's time to stop the training.</p>
<p>But we observe that except for coherence, the value of other measures stops changing quite early in the training process. Hence, coherence could prove to be quite useful in this case as it easily captures the change in topic distributions over increasing epochs and indicate correctly when the topics stop changing much in the training process.</p>
<hr />
<h1>References</h1>
<h2>Papers</h2>
<ul>
<li>Chang et. al <em>Reading Tea Leaves: How Humans Interpret Topic Models</em>, NIPS 2009</li>
<li>Wallach et. al <em>Evaluation Methods for Topic Models</em>, ICML 2009</li>
<li>Lau et. al <em>Machine Reading Tea Leaves: Automatically Evaluating Topic Coherence and Topic Model Quality</em>, ACL 2014</li>
<li>
<p>R&ouml;der et. al <em>Exploring the Space of Topic Coherence Methods</em>, Web Search and Data Mining 2015</p>
</li>
<li>
<p>Sievert et. al <em>LDAvis: A method for visualizing and interpreting topics</em> ACL 2014 Workshop on Interactive Language Learning, Visualization, and Interfaces</p>
</li>
<li>
<p>Chuang et. al <em>Termite: Visualization Techniques for Assessing Textual Topic Models</em>, AVI 2012 <a href="http://vis.stanford.edu/papers/termite">link</a></p>
</li>
<li>Chuang et. al <em>Topic Model Diagnostics: Assessing Domain Relevance via Topical Alignment</em>, ICML 2013 <a href="http://vis.stanford.edu/papers/topic-model-diagnostics">link</a></li>
</ul>
<h2>Software</h2>
<ul>
<li><a href="http://radimrehurek.com/gensim">gensim Topic Modelling for Humans</a> (Python)</li>
<li><a href="http://mallet.cs.umass.edu/">UMass Machine Learning for Language - Mallet</a> (Java)</li>
<li><a href="https://nlp.stanford.edu/software/tmt/tmt-0.3/">Stanford Topic Modelling Toolbox</a> (Java)</li>
<li><a href="https://github.com/Ardavans/sHDP">Spherical Hierarchical Dirichlet Processes</a></li>
<li>Termite
<ul>
<li><a href="https://github.com/StanfordHCI/termite">Original project</a></li>
<li><a href="https://github.com/uwdata/termite-data-server">Data server</a></li>
<li><a href="https://github.com/uwdata/termite-visualizations">Visualisation</a></li>
</ul>
</li>
<li>scattertext
<ul>
<li>scattertext allows you to plot differential word usage patterns from two corpora into an interactive display. It's not exactly an evaluation method for topic models but can be quite useful for analysing corpora</li>
<li>there's a talk by the creator at PyData Seattle 2017 <a href="https://pydata.org/seattle2017/schedule/presentation/69/">link</a></li>
</ul>
</li>
</ul>
<h2>Datasets</h2>
<p>The model used in this notebook is built on the Kaggle Fake News dataset available <a href="https://www.kaggle.com/mrisdal/fake-news">here</a>.</p>
<h2>Interwebs</h2>
<ul>
<li><a href="http://qpleple.com/perplexity-to-evaluate-topic-models/">http://qpleple.com/perplexity-to-evaluate-topic-models/</a></li>
<li><a href="http://qpleple.com/topic-coherence-to-evaluate-topic-models/">http://qpleple.com/topic-coherence-to-evaluate-topic-models/</a></li>
</ul>
<h2>General stuff about NLP you might be interested in</h2>
<ul>
<li>Yoav Goldberg on evaluating NNLMs
<ul>
<li><a href="https://medium.com/@yoav.goldberg/an-adversarial-review-of-adversarial-generation-of-natural-language-409ac3378bd7">Original post</a></li>
<li><a href="https://medium.com/@yoav.goldberg/clarifications-re-adversarial-review-of-adversarial-learning-of-nat-lang-post-62acd39ebe0d">Addendum</a></li>
<li><a href="https://www.facebook.com/yann.lecun/posts/10154498539442143">Yann le Cunn on the matter</a></li>
<li><a href="https://medium.com/@yoav.goldberg/a-response-to-yann-lecuns-response-245125295c02">A response to le Cunn</a></li>
</ul>
</li>
</ul>
