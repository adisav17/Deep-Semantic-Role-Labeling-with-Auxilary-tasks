# Deep-Semantic-Role-Labeling-with-Auxiliary-tasks

 
We analyze the effects of incorporation of linguistic features on the performance and generalization abilities of our model.
Sample notebooks displayed for demonstration. 
For the entire partitive group Nombank task, For our best Bert-based model with predicate indicator embeddings and positional embeddings attached to it, we get an F-score of 88.7. We call this model 1. For our best Bert-based model trained with an auxiliary model tuning on linguistic features, we get an F-score of 89.2. We call this model 2. 

Given a model 1 and model 2 trained on the partitive task, we make the predict on the percent task. We get an F-score of 90.6 with model 1. We get an F-score of 96.1 with model 2. If we use rapid fine tuning, re-training the models for a few epochs on the percent task, we get an F-score of 92.3 with model 1. We get an F-score of 97.2 with model 2
We evaluate the out-of-distribution generalization performances our our auxiliary model in comparision to our Bert base fine-tuned model. When using model 1 for relational nouns, we get an F-score of 77.6.
When using model 2 for relational nouns, we get an F-score of 81.3.
