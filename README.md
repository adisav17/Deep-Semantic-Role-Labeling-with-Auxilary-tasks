# Deep-Semantic-Role-Labeling-with-Auxiliary-tasks
Semantic role labeling with a bert based model integrated with linguistic features. Linguistic features seen to help generalization in natural language inference and help the F-score.
For the entire partitive group Nombank task,
For the bert based model with predicate indicator embeddings and positional embeddings attached to it, we get an F-score of 83.8. We call this model 1.
For the bert based model trained with an auxiliary model tuning on linguistic features, we get an F-score of 84.6. We call this model 2.
For the multi view ensemble model, we get an F-score of 87.2. 

This is equivalent to the _score_file_without_adjustment_ in the scoring python script. 

With regards to generalization testing, 

Given a model 1 and model 2 trained on the partitive task,
We use zero shot shot learning to predict on the percent task. 
We get an F-score of 90.6 with model 1.
We get an F-score of 96.1 with model 2.  


We use zero shot shot learning to predict on the percent task. 
We get an F-score of 92.3 with model 1.
We get an F-score of 97.2 with model 2.
